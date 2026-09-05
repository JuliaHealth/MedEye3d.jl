using Downloads
body = IOBuffer("{\"test\":\"data\"}")
resp = Downloads.request("https://httpbin.org/post", method="POST", input=body, headers=["Content-Type" => "application/json"])
println(resp.status)
