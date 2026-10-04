function changeClip(minVal, maxVal, value, color, range)
    if value < minVal
        return 0.0
    elseif value >= maxVal
        return color
    else
        return color * ((value - minVal) / max(range, 0.001))
    end
end

PETRes = 0.0
PET_Min = 0.1
PET_Max = 10.0
PETValueRange = max(PET_Max - PET_Min, 1.0)
PETColorMask = (r=1.0, g=0.5, b=0.0)

if PET_Min == PET_Max
    if abs(PETRes - PET_Min) < 0.1
        println("petOnly = vec3(1.0, 0.5, 0.0)")
    else
        println("petOnly = vec3(0.0)")
    end
elseif PETRes > PET_Min
    normalizedVal = clamp((PETRes - PET_Min) / max(PETValueRange, 0.001), 0.0, 1.0)
    if normalizedVal > 0.0
        intensity = normalizedVal^0.4
        maskColor = (
            changeClip(PET_Min, PET_Max, PETRes, PETColorMask.r, PETValueRange),
            changeClip(PET_Min, PET_Max, PETRes, PETColorMask.g, PETValueRange),
            changeClip(PET_Min, PET_Max, PETRes, PETColorMask.b, PETValueRange)
        )
        petOnly = (
            clamp(maskColor[1] * intensity * 1.8, 0.0, 1.0),
            clamp(maskColor[2] * intensity * 1.8, 0.0, 1.0),
            clamp(maskColor[3] * intensity * 1.8, 0.0, 1.0)
        )
        println("petOnly = ", petOnly)
    else
        println("petOnly = vec3(0.0) (normalizedVal <= 0)")
    end
else
    println("petOnly = vec3(0.0) (PETRes <= PET_Min)")
end
