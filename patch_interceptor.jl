content = read("src/display/LesionMetadataWindow.jl", String)

# Remove the bad print statements
content = replace(content, r"    println\(\"\[DEBUG BBOX\].*?\n.*?flush\(stdout\)\n"s => "")

new_interceptor = """    # ── Intercept Clicks to Prevent Click-Through ─────────────────────────────
    on(events(fig.scene).mousebutton, priority=100) do event
        if !_login_authenticated[]
            if event.action == Mouse.press
                pos = events(fig.scene).mouseposition[]
                
                # Check Textboxes
                user_bbox = login_tb_user.layoutobservables.computedbbox[]
                pass_bbox = login_tb_pass.layoutobservables.computedbbox[]
                btn_bbox = login_btn.layoutobservables.computedbbox[]
                
                # Function to check if pos is in bbox
                in_rect(p, b) = (p[1] >= b.origin[1] && p[1] <= b.origin[1] + b.widths[1] &&
                                 p[2] >= b.origin[2] && p[2] <= b.origin[2] + b.widths[2])
                
                if in_rect(pos, btn_bbox)
                    login_btn.buttoncolor = RGBf(0.1, 0.4, 0.2)
                    _do_login()
                    return Consume(true)
                elseif in_rect(pos, user_bbox) || in_rect(pos, pass_bbox)
                    return Consume(false) # Let the Textboxes (priority 60) handle it!
                else
                    return Consume(true) # Block everything else (like Menus at 64)
                end
            elseif event.action == Mouse.release
                login_btn.buttoncolor = RGBf(0.2, 0.6, 0.3)
            end
        elseif time() - _login_success_time[] < 1.0
            return Consume(true)
        end
        return Consume(false)
    end"""

content = replace(content, r"    # ── Intercept Clicks to Prevent Click-Through ─────────────────────────────.*?return Consume\(false\)\n    end"s => new_interceptor)

write("src/display/LesionMetadataWindow.jl", content)
println("Patched interceptor!")
