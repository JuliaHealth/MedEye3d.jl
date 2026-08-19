for i in 1:15
    try
        conn = connect("127.0.0.1", 5005)
        close(conn)
        println("Connected!")
        break
    catch e
        println("Failed to connect, retrying...")
        sleep(1)
    end
end
