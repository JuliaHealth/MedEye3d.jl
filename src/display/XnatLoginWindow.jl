"""
    XnatLoginWindow

Dedicated, separate window for MedEye3D XNAT Authentication.
Decoupled completely from the main Makie viewer window to prevent event conflicts,
layout solver issues, or GUI blocking.
"""
module XnatLoginWindow

using GLMakie
import GLMakie.Makie as Makie
using Observables
import Dates
import ..XnatAuth
import ..SegmentationDisplay: synchronized_makie_renderloop, GLOBAL_OPENGL_LOCK

export open_xnat_login_window, close_xnat_login_window, is_login_window_open,
       active_login_screen, active_login_fig, login_status_obs, login_status_color

# ── Global Window State ──────────────────────────────────────────────────────
const active_login_screen = Ref{Any}(nothing)
const active_login_fig = Ref{Any}(nothing)
const login_status_obs = Observable{String}("")
const login_status_color = Observable{RGBf}(RGBf(0.85, 0.88, 0.92))
const on_login_success = Ref{Any}(nothing)
const on_login_close = Ref{Any}(nothing)
const login_tb_user_ref = Ref{Any}(nothing)
const login_tb_pass_ref = Ref{Any}(nothing)

"""
    is_login_window_open() -> Bool

Returns true if the separate login window is currently open and alive.
"""
function is_login_window_open()::Bool
    scr = active_login_screen[]
    scr === nothing && return false
    try
        return isopen(scr)
    catch
        return false
    end
end

"""
    close_xnat_login_window()

Safely closes the separate XNAT login window if open.
"""
function close_xnat_login_window()
    scr = active_login_screen[]
    active_login_screen[] = nothing
    active_login_fig[] = nothing
    cb_close = on_login_close[]
    on_login_close[] = nothing
    on_login_success[] = nothing

    if scr !== nothing
        println("[LOGIN] Closing companion login window..."); flush(stdout)
        try
            scr.stop_renderloop[] = true
            if scr.owns_glscreen && GLMakie.GLAbstraction.context_alive(scr.glscreen)
                GLMakie.GLFW.HideWindow(scr.glscreen)
                GLMakie.GLFW.SetWindowShouldClose(scr.glscreen, true)
            end
        catch e
            @warn "Failed closing XNAT login window: $e"
        end
        @async begin
            try
                scr.stop_renderloop[] = true
                if scr.rendertask !== nothing && Base.current_task() != scr.rendertask
                    try wait(scr.rendertask) catch; end
                end
                scr.rendertask = nothing
                lock(GLOBAL_OPENGL_LOCK) do
                    close(scr; reuse=false)
                end
                println("[LOGIN] Companion login window closed and cleaned up successfully"); flush(stdout)
            catch e
                @debug "Async screen close cleanup: $e"
            end
        end
    end
    if cb_close !== nothing
        try cb_close() catch e; @error "Login on_close callback error: $e"; end
    end
end

"""
    open_xnat_login_window(; on_success=nothing, on_close=nothing) -> Union{GLMakie.Screen, Nothing}

Opens (or focuses/brings to front) the dedicated XNAT login window.
"""
function open_xnat_login_window(; on_success=nothing, on_close=nothing)
    if on_success !== nothing
        on_login_success[] = on_success
    end
    if on_close !== nothing
        on_login_close[] = on_close
    end

    # If already open and active, bring to front and focus
    if is_login_window_open()
        scr = active_login_screen[]
        try
            lock(GLOBAL_OPENGL_LOCK) do
                if scr.owns_glscreen && GLMakie.GLAbstraction.context_alive(scr.glscreen)
                    GLMakie.GLFW.RestoreWindow(scr.glscreen)
                    GLMakie.GLFW.ShowWindow(scr.glscreen)
                    GLMakie.GLFW.FocusWindow(scr.glscreen)
                end
            end
            return scr
        catch; end
    end

    # Load previously cached credentials for convenient pre-fill
    cached = XnatAuth.load_cached_credentials()
    init_url = (cached !== nothing && !isempty(cached.server_url)) ? cached.server_url : XnatAuth.DEFAULT_XNAT_SERVER
    init_alias = (cached !== nothing) ? cached.alias : ""
    init_secret = (cached !== nothing && XnatAuth.is_token_valid_locally(cached)) ? cached.secret : ""
    init_user = (cached !== nothing) ? cached.username : ""

    # Theme colors matching MedEye3D
    BG_WIN = RGBf(0.12, 0.13, 0.16)
    BG_TB  = RGBf(0.18, 0.20, 0.26)
    BG_TB_FOC = RGBf(0.24, 0.28, 0.38)
    BORDER_TB = RGBf(0.35, 0.40, 0.50)
    TXT_CLR = RGBf(0.95, 0.95, 0.95)
    SUBTXT_CLR = RGBf(0.65, 0.70, 0.78)

    fig = Figure(
        size = (480, 420),
        backgroundcolor = BG_WIN,
        figure_padding = (25, 25, 20, 20)
    )

    # ── Header ───────────────────────────────────────────────────────────────
    Label(fig[1, 1:2], "MedEye3D — XNAT Authentication",
        fontsize = 18, font = :bold, color = :white, halign = :center)

    Label(fig[2, 1:2], "Enter 24h Alias & Secret, or Username & Password to issue token",
        fontsize = 10, color = SUBTXT_CLR, halign = :center)

    # Status / Error message label
    if cached !== nothing && XnatAuth.is_token_valid_locally(cached)
        time_left_sec = cached.expires_at - time()
        hrs = max(0, Int(floor(time_left_sec / 3600)))
        mins = max(0, Int(floor(mod(time_left_sec, 3600) / 60)))
        uname = isempty(cached.username) ? cached.alias : cached.username
        login_status_obs[] = "Active session for '$(uname)' ($(hrs)h $(mins)m remaining)"
        login_status_color[] = RGBf(0.40, 0.90, 0.40)
    elseif cached !== nothing && !XnatAuth.is_token_valid_locally(cached)
        login_status_obs[] = "Previous token expired. Please enter new credentials."
        login_status_color[] = RGBf(0.95, 0.75, 0.30)
    else
        login_status_obs[] = ""
        login_status_color[] = RGBf(0.95, 0.75, 0.30)
    end
    Label(fig[3, 1:2], login_status_obs,
        fontsize = 11, color = login_status_color, halign = :center, tellwidth = false)

    # ── Input Fields Grid ────────────────────────────────────────────────────
    g = GridLayout(fig[4, 1:2])
    rowgap!(g, 8)

    # Row 1: URL
    Label(g[1, 1], "Server URL:", fontsize = 12, color = :white, halign = :right)
    tb_url = Textbox(g[1, 2], placeholder = "XNAT Server URL",
        stored_string = init_url, displayed_string = init_url, fontsize = 12,
        textcolor = TXT_CLR, boxcolor = BG_TB, boxcolor_focused = BG_TB_FOC,
        bordercolor = BORDER_TB, cursorcolor = :white)

    # Row 2: 24h Alias
    Label(g[2, 1], "24h Alias:", fontsize = 12, color = :white, halign = :right)
    tb_alias = Textbox(g[2, 2], placeholder = "Paste 24h Alias Token",
        stored_string = init_alias, displayed_string = init_alias, fontsize = 12,
        textcolor = TXT_CLR, boxcolor = BG_TB, boxcolor_focused = BG_TB_FOC,
        bordercolor = BORDER_TB, cursorcolor = :white)

    # Row 3: 24h Secret
    Label(g[3, 1], "24h Secret:", fontsize = 12, color = :white, halign = :right)
    tb_secret = Textbox(g[3, 2], placeholder = "Paste 24h Secret Token",
        stored_string = init_secret, displayed_string = init_secret, fontsize = 12,
        textcolor = TXT_CLR, boxcolor = BG_TB, boxcolor_focused = BG_TB_FOC,
        bordercolor = BORDER_TB, cursorcolor = :white)

    # Row 4: Username
    Label(g[4, 1], "Username:", fontsize = 12, color = :white, halign = :right)
    tb_user = Textbox(g[4, 2], placeholder = "Enter username (optional)",
        stored_string = init_user, displayed_string = init_user, fontsize = 12,
        textcolor = TXT_CLR, boxcolor = BG_TB, boxcolor_focused = BG_TB_FOC,
        bordercolor = BORDER_TB, cursorcolor = :white)
    login_tb_user_ref[] = tb_user

    # Row 5: Password
    Label(g[5, 1], "Password:", fontsize = 12, color = :white, halign = :right)
    tb_pass = Textbox(g[5, 2], placeholder = "Enter password (optional)",
        fontsize = 12, textcolor = TXT_CLR, boxcolor = BG_TB, boxcolor_focused = BG_TB_FOC,
        bordercolor = BORDER_TB, cursorcolor = :white)
    login_tb_pass_ref[] = tb_pass
    colsize!(g, 1, Fixed(100))
    colsize!(g, 2, Auto())

    # ── Action Buttons ───────────────────────────────────────────────────────
    btn_g = GridLayout(fig[5, 1:2])
    colgap!(btn_g, 15)
    btn_connect = Button(btn_g[1, 1], label = "Login / Connect",
        buttoncolor = RGBf(0.20, 0.55, 0.28), labelcolor = :white, fontsize = 13, height = 36)
    btn_offline = Button(btn_g[1, 2], label = "Work Offline / Close",
        buttoncolor = RGBf(0.28, 0.35, 0.45), labelcolor = :white, fontsize = 13, height = 36)

    rowsize!(fig.layout, 1, Fixed(28))
    rowsize!(fig.layout, 2, Fixed(18))
    rowsize!(fig.layout, 3, Fixed(22))
    rowsize!(fig.layout, 4, Auto())
    rowsize!(fig.layout, 5, Fixed(44))

    # ── Login Logic ──────────────────────────────────────────────────────────
    _is_executing_login = Ref(false)
    function _execute_login()
        if _is_executing_login[]
            println("[LOGIN] Login request already in progress, ignoring duplicate trigger"); flush(stdout)
            return
        end
        get_str(tb) = begin
            disp = tb.displayed_string[]
            stor = tb.stored_string[]
            s_disp = disp === nothing ? "" : String(strip(string(disp)))
            s_stor = stor === nothing ? "" : String(strip(string(stor)))
            isempty(s_disp) ? s_stor : s_disp
        end
        url = get_str(tb_url)
        (isempty(url) || occursin("bmia.nl", url)) && (url = XnatAuth.DEFAULT_XNAT_SERVER)
        alias = get_str(tb_alias)
        secret = get_str(tb_secret)
        user = get_str(tb_user)
        pass = get_str(tb_pass)

        println("[LOGIN] Login triggered (url='$url', user='$user', has_pass=$(!isempty(pass)), alias='$(isempty(alias) ? "" : alias[1:min(8,length(alias))])...', has_secret=$(!isempty(secret)))"); flush(stdout)

        # Priority 1: Offline bypass / testing password
        if pass == "lu" || pass == "offline"
            println("[LOGIN] Priority 1: Authenticating in offline/bypass mode"); flush(stdout)
            login_status_obs[] = "Authenticated locally (offline mode)"
            login_status_color[] = RGBf(0.40, 0.90, 0.40)
            cb = on_login_success[]
            if cb !== nothing
                try cb(nothing; offline=true) catch e; @error "Login callback error: $e"; end
            end
            close_xnat_login_window()
            return
        end

        # Priority 2: Direct Username & Password login -> Issues fresh 24h token from XNAT
        if !isempty(user) && !isempty(pass)
            _is_executing_login[] = true
            println("[LOGIN] Priority 2: Connecting to XNAT and issuing 24h token for user '$user'..."); flush(stdout)
            login_status_obs[] = "Connecting to XNAT and issuing 24h token..."
            login_status_color[] = RGBf(0.40, 0.80, 1.00)
            @async begin
                try
                    tok_creds, tok_msg = try
                        XnatAuth.issue_token_from_userpass(url, user, pass)
                    catch e
                        (nothing, "Connection error: $e")
                    end
                    println("[LOGIN] Issue token response: $(tok_creds !== nothing ? "SUCCESS" : "FAILED") ($tok_msg)"); flush(stdout)
                    if tok_creds !== nothing
                        XnatAuth.save_cached_credentials(tok_creds)
                        login_status_obs[] = "Success! Issued token for '$(tok_creds.username)'"
                        login_status_color[] = RGBf(0.40, 0.90, 0.40)
                        cb = on_login_success[]
                        if cb !== nothing
                            try cb(tok_creds; offline=false) catch e; @error "Login callback error: $e"; end
                        end
                        close_xnat_login_window()
                    else
                        login_status_color[] = RGBf(1.00, 0.30, 0.30)
                        if occursin("Connection", tok_msg) || occursin("502", tok_msg) || occursin("503", tok_msg)
                            login_status_obs[] = "XNAT in maintenance or unreachable. You can Work Offline."
                        else
                            login_status_obs[] = !isempty(tok_msg) ? tok_msg : "Authentication failed with provided credentials"
                        end
                    end
                finally
                    _is_executing_login[] = false
                end
            end
            return
        end

        # Priority 3: 24h Rotating Alias & Secret Token
        if !isempty(alias) && !isempty(secret)
            _is_executing_login[] = true
            println("[LOGIN] Priority 3: Validating 24h token for user '$user' against '$url'..."); flush(stdout)
            now_t = time()
            cached_prev = XnatAuth.load_cached_credentials()
            issued_t = (cached_prev !== nothing && cached_prev.alias == alias && cached_prev.secret == secret) ?
                        cached_prev.issued_at : now_t
            exp_t = issued_t + 86400.0

            new_creds = XnatAuth.XnatCredentials(
                server_url = url,
                alias = alias,
                secret = secret,
                username = !isempty(user) ? user : alias,
                issued_at = issued_t,
                expires_at = exp_t
            )
            login_status_obs[] = "Validating token with XNAT..."
            login_status_color[] = RGBf(0.40, 0.80, 1.00)
            @async begin
                try
                    is_valid, val_msg = try
                        XnatAuth.validate_credentials(new_creds)
                    catch e
                        (false, "Connection error: $e")
                    end
                    println("[LOGIN] Token validation response: $(is_valid ? "SUCCESS" : "FAILED") ($val_msg)"); flush(stdout)
                    if is_valid
                        XnatAuth.save_cached_credentials(new_creds)
                        login_status_obs[] = "Success! Authenticated via 24h token"
                        login_status_color[] = RGBf(0.40, 0.90, 0.40)
                        cb = on_login_success[]
                        if cb !== nothing
                            try cb(new_creds; offline=false) catch e; @error "Login callback error: $e"; end
                        end
                        close_xnat_login_window()
                    else
                        login_status_color[] = RGBf(1.00, 0.30, 0.30)
                        if occursin("Connection", val_msg) || occursin("502", val_msg) || occursin("503", val_msg)
                            login_status_obs[] = "XNAT in maintenance or unreachable. You can Work Offline."
                        else
                            login_status_obs[] = !isempty(val_msg) ? val_msg : "Invalid Alias/Secret token for $url"
                        end
                        tb_secret.displayed_string[] = ""
                        tb_secret.stored_string[] = ""
                    end
                finally
                    _is_executing_login[] = false
                end
            end
            return
        end

        println("[LOGIN] Insufficient credentials: need alias & secret or user & pass"); flush(stdout)
        login_status_color[] = RGBf(1.00, 0.30, 0.30)
        login_status_obs[] = "Please provide 24h Alias & Secret, or Username & Password"
    end

    on(btn_connect.clicks) do _
        _execute_login()
    end

    on(btn_offline.clicks) do _
        println("[LOGIN] 'Work Offline / Close' clicked"); flush(stdout)
        cb = on_login_success[]
        if cb !== nothing
            try cb(nothing; offline=true) catch e; @error "Login callback error: $e"; end
        end
        close_xnat_login_window()
    end

    on(tb_pass.stored_string)   do _; _execute_login(); end
    on(tb_user.stored_string)   do _
        pass_str = String(strip(string(tb_pass.displayed_string[])))
        if !isempty(pass_str)
            _execute_login()
        end
    end
    on(tb_secret.stored_string) do _; _execute_login(); end
    on(tb_alias.stored_string)  do _; _execute_login(); end

    # Escape key dismisses window
    on(events(fig).keyboardbutton) do event
        if (event.action == Makie.Keyboard.release || event.action == Makie.Keyboard.press) && event.key == Makie.Keyboard.escape
            println("[LOGIN] Escape pressed, closing companion login window"); flush(stdout)
            close_xnat_login_window()
            return Consume(true)
        end
        return Consume(false)
    end

    # Display independent screen using synchronized renderloop
    screen = lock(GLOBAL_OPENGL_LOCK) do
        s = GLMakie.Screen(;
            renderloop = synchronized_makie_renderloop,
            title = "MedEye3D — XNAT Authentication",
            focus_on_show = true
        )
        display(s, fig)
        s
    end

    active_login_screen[] = screen
    active_login_fig[] = fig

    try
        if screen.owns_glscreen && GLMakie.GLAbstraction.context_alive(screen.glscreen)
            GLMakie.GLFW.SetWindowCloseCallback(screen.glscreen) do _
                close_xnat_login_window()
            end
        end
    catch; end

    println("[XNAT] Separate login window opened (glscreen: $(screen.glscreen))"); flush(stdout)
    return screen
end

end # module XnatLoginWindow
