#version 450
layout(set = 1, binding = 0) uniform TextureParams {
    int   CTisVisible;
    float CTminValue;
    float CTmaxValue;
    float CTValueRange;
    float CTmaskContribution;
    vec4  CTColorMask;
    int   CTallowedIDCount;
    float CTallowedIDs[16];
    
    int   PETisVisible;
    float PETminValue;
    float PETmaxValue;
    float PETValueRange;
    float PETmaskContribution;
    vec4  PETColorMask;
    int   PETallowedIDCount;
    float PETallowedIDs[16];
} params;
layout(location = 0) out vec4 outColor;
void main() {
    outColor = vec4(params.CTisVisible + params.PETisVisible);
}
