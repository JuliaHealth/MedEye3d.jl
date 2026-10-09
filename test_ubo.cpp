#include <iostream>
#include <cstddef>
struct TextureParams {
    int   CTisVisible;           // 0
    float CTminValue;            // 4
    float CTmaxValue;            // 8
    float CTValueRange;          // 12
    float CTmaskContribution;    // 16
    alignas(16) float CTColorMask[4]; // 32
    int   CTallowedIDCount;      // 48
    alignas(16) float CTallowedIDs[16]; // 64 (array size 256, ends 320)
    
    int   PETisVisible;          // 320 ?
    float PETminValue;           // 324
    float PETmaxValue;           // 328
    float PETValueRange;         // 332
    float PETmaskContribution;   // 336
    alignas(16) float PETColorMask[4]; // 352 ?
};
int main() {
    std::cout << "PETisVisible: " << offsetof(TextureParams, PETisVisible) << "\n";
    std::cout << "PETminValue: " << offsetof(TextureParams, PETminValue) << "\n";
    std::cout << "PETColorMask: " << offsetof(TextureParams, PETColorMask) << "\n";
    return 0;
}
