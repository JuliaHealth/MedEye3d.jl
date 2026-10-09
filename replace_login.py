import re

with open("src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

# We need to find the start of "# ── Login Overlay ───"
start_str = "    # ── Login Overlay ─────────────────────────────────────────────────────────\n"
end_str = "    lock(GLOBAL_OPENGL_LOCK) do\n"

start_idx = content.find(start_str)
end_idx = content.find(end_str, start_idx)

if start_idx == -1 or end_idx == -1:
    print("Could not find start or end index!")
    exit(1)

new_login_code = """    # ── Separate Login Window ──────────────────────────────────────────────────
    _login_visible = Observable(false)
    _lmw_observables[:login_visible] = _login_visible
    
    # Hide all dropdowns while login is active
    on(_login_visible) do vis
        for m in _ALL_MENUS
            try
                if hasproperty(m, :blockscene)
                    m.blockscene.visible[] = !vis
                end
            catch
            end
        end
        
        if vis && !_login_authenticated[]
            # Create the separate login window
            login_fig = Figure(size = (350, 200), backgroundcolor = RGBf(0.12, 0.14, 0.18))
            
            login_inner = GridLayout(login_fig[1, 1], halign=:center, valign=:center)
            
            lbl_title = Label(login_inner[1, 1:2], "MedEye3d Login", color=:white, fontsize=24, font=:bold, halign=:center)
            lbl_user = Label(login_inner[2, 1], "Username:", color=:white, fontsize=14, halign=:right)
            login_tb_user = Textbox(login_inner[2, 2], placeholder="Enter username", fontsize=14, width=200,
                textcolor=RGBf(0.95, 0.95, 0.95), textcolor_placeholder=RGBf(0.55, 0.58, 0.65), 
                boxcolor=RGBf(0.18, 0.20, 0.25), boxcolor_focused=RGBf(0.25, 0.28, 0.35), 
                bordercolor=RGBf(0.35, 0.40, 0.50), cursorcolor=RGBf(0.95, 0.95, 0.95))
            
            lbl_pass = Label(login_inner[3, 1], "Password:", color=:white, fontsize=14, halign=:right)
            login_tb_pass = Textbox(login_inner[3, 2], placeholder="Enter password", fontsize=14, width=200,
                textcolor=RGBf(0.95, 0.95, 0.95), textcolor_placeholder=RGBf(0.55, 0.58, 0.65), 
                boxcolor=RGBf(0.18, 0.20, 0.25), boxcolor_focused=RGBf(0.25, 0.28, 0.35), 
                bordercolor=RGBf(0.35, 0.40, 0.50), cursorcolor=RGBf(0.95, 0.95, 0.95))
            
            login_btn = Button(login_inner[4, 1:2], label="Login", buttoncolor=RGBf(0.2, 0.6, 0.3), labelcolor=:white, fontsize=16)
            login_msg = Label(login_inner[5, 1:2], "", color=RGBf(1.0, 0.3, 0.3), fontsize=12, halign=:center)
            
            rowsize!(login_inner, 1, Fixed(40))
            rowsize!(login_inner, 2, Fixed(35))
            rowsize!(login_inner, 3, Fixed(35))
            rowsize!(login_inner, 4, Fixed(40))
            rowsize!(login_inner, 5, Fixed(25))
            colsize!(login_inner, 1, Fixed(100))
            colsize!(login_inner, 2, Fixed(210))
            
            _lmw_observables[:login_fig] = login_fig
            _lmw_observables[:login_user] = login_tb_user
            
            login_screen = lock(GLOBAL_OPENGL_LOCK) do
                s = GLMakie.Screen(login_fig.scene; renderloop=synchronized_makie_renderloop)
                display(s, login_fig)
                s
            end
            _lmw_observables[:login_screen] = login_screen
            
            # Make sure we track focus
            events(login_fig.scene).hasfocus[] = true
            
            function _do_login()
                user = strip(login_tb_user.displayed_string[])
                pass = strip(login_tb_pass.displayed_string[])
                
                if isempty(user)
                    login_msg.text[] = "Please enter a username"
                    return
                end
                if pass != _LOGIN_PASSWORD
                    login_msg.text[] = "Incorrect password"
                    return
                end
                _current_user[] = user
                _login_authenticated[] = true
                _login_success_time[] = time()
                
                _login_visible[] = false
                println("[LOGIN] User  authenticated at $(Dates.format(Dates.now(), "HH:MM:SS"))"); flush(stdout)
                
                # Close the window
                try
                    GLMakie.close(login_screen)
                catch
                end
            end
            
            on(login_btn.clicks) do _
                _do_login()
            end
            on(login_tb_pass.stored_string) do _
                _do_login()
            end
            on(login_tb_user.stored_string) do _
                _do_login()
            end
        end
    end

    # Block clicks on the main figure if login is not authenticated
    on(events(fig.scene).mousebutton, priority=100) do event
        if !_login_authenticated[]
            return Consume(true) # Block ALL interactions on the main window
        elseif time() - _login_success_time[] < 1.0
            return Consume(true)
        end
        return Consume(false)
    end

"""

new_content = content[:start_idx] + new_login_code + content[end_idx:]

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(new_content)

print("Replacement successful!")
