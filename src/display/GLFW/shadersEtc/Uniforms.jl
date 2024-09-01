
"""
managing  uniform values - global values in shaders
"""
module Uniforms
using StaticArrays, ModernGL, Dictionaries, Parameters, ColorTypes
using ..ForDisplayStructs

export isMaskDiffViss, changeMainTextureContribution, changeTextureContribution, coontrolMinMaxUniformVals, createStructsDict, setCTWindow, setMaskColor, setTextureVisibility, setTypeOfMainSampler!
export @uniforms
export @uniforms!


########## this from https://github.com/jorge-brito/Glutils.jl/blob/4b7da5895a3e927792994ad0b6643498a8125362/src/utils.jl
struct TypeRef{T} end

Base.adjoint(x::Ref) = x[]
Base.adjoint(::Type{T}) where {T} = TypeRef{T}()
Base.getindex(::TypeRef{T}) where {T} = Ref{T}()
Base.getindex(::TypeRef{T}, x::S) where {T,S} = Ref{T}(x)

const SymString = Union{AbstractString,Symbol}

macro map(expr...)
    T, ex =
        if length(expr) == 1
            nothing, first(expr)
        else
            first(expr), last(expr)
        end

    args = map(ex.args) do arg
        if Meta.isexpr(arg, :(=))
            key, value = arg.args
            return :($(QuoteNode(key)) => $value)
        elseif Meta.isexpr(arg, :(:=))
            key, value = arg.args
            return :($key => $value)
        else
            error("Invalid syntax for @map, expected expression of type `foo = bar` or `foo := bar`")
        end
    end

    result =
        if isnothing(T)
            Expr(:call, :Dict, args...)
        else
            Expr(:call, Expr(:curly, :Dict, T.args...), args...)
        end

    return esc(result)
end

const GL_ENUM_TYPE = @map Symbol, GLenum {
    GLboolean = GL_BOOL,
    GLchar = GL_BYTE,
    GLdouble = GL_DOUBLE,
    GLfloat = GL_FLOAT,
    GLint = GL_INT,
    GLshort = GL_SHORT,
    GLubyte = GL_UNSIGNED_BYTE,
    GLuint = GL_UNSIGNED_INT,
    GLushort = GL_UNSIGNED_SHORT,
}

macro genumof(type)
    :(GL_ENUM_TYPE[$(QuoteNode(type))])
end

genumof(::Type{Cint}) = GL_INT
genumof(::Type{Cchar}) = GL_BYTE
genumof(::Type{Cfloat}) = GL_FLOAT
genumof(::Type{Cshort}) = GL_SHORT
genumof(::Type{Int64}) = GL_DOUBLE
genumof(::Type{Cuint}) = GL_UNSIGNED_INT
genumof(::Type{Cuchar}) = GL_UNSIGNED_BYTE
genumof(::Type{Cushort}) = GL_UNSIGNED_SHORT

gbool(x::GLenum) = x == GL_TRUE
gbool(x::Bool) = x ? GL_TRUE : GL_FALSE

"""
        @g_str(name) -> GLenum
Returns a OpenGL constant.
# Examples
```julia
g"bool" == GL_BOOL
g"Texture 2D" == GL_TEXTURE_2D
g"Clamp.to.edge" == GL_CLAMP_TO_EDGE
g"texture-wrap-t" == GL_TEXTURE_WRAP_T
```
"""
macro g_str(ex)
    name = uppercase(replace(ex, r"(\s|\.|\-)+" => "_"))
    esc(Symbol("GL_$name"))
end



############# part below directly copied from https://github.com/jorge-brito/Glutils.jl/blob/master/src/uniforms.jl

# this function is used to get the
# suffix of the glUniform function
# for the correct type
_type_suffix(::Type{<:Bool}) = "i"
_type_suffix(::Type{<:Signed}) = "i"
_type_suffix(::Type{<:Unsigned}) = "ui"
_type_suffix(::Type{<:AbstractFloat}) = "f"

"""
        uniform!(location, values...) -> Nothing
Set the uniform `vec` variable at `location`.
# Examples
```julia
myuniform = getuniform(program, "someUniform")
# set float vec3
uniform!(myuniform, 1.0, 1.0, 0.5)
# set float vec4
uniform!(myuniform, Cfloat[1, 2, 3, 4])
# set a 4x4 matrix
uniform!(myuniform, rand(Cfloat, 4, 4))
```
"""
function uniform! end

@generated function uniform!(location, values::Vararg{T,N}) where {N,T<:Real}
    suffix = _type_suffix(T)
    glFunc = Symbol("glUniform$(N)$suffix")
    return :($(glFunc)(location, values...))
end

@generated function uniform!(location, vector::SVector{N,T}) where {N,T<:Real}
    suffix = _type_suffix(T)
    glFunc = Symbol("glUniform$(N)$(suffix)v")
    return :($(glFunc)(location, 1, vector))
end

@generated function uniform!(location, matrix::SMatrix{N,M,T}, transposed::Bool=false) where {N,M,T<:Real}
    glFunc = Symbol(
        if N == M
            "glUniformMatrix$(N)fv"
        else
            "glUniformMatrix$(N)x$(M)fv"
        end
    )
    return :($(glFunc)(location, 1, transposed, Cfloat[matrix...]))
end

function uniform!(location, vector::AbstractVector{T}) where {T<:Real}
    N = length(vector)
    uniform!(location, SVector{N,T}(vector...))
end

function uniform!(location, matrix::AbstractMatrix{T}, transposed::Bool=false) where {T<:Real}
    N, M = size(matrix)
    uniform!(location, SMatrix{N,M,T}(matrix), transposed)
end
"""
        getuniform(program, name) -> GLint
Gets the location of the uniform variable
identified by `name`.
"""
getuniform(program, name::SymString) = glGetUniformLocation(program, string(name))
"""
        @uniforms foo, bar, ... = program
Get the location of `foo`, `bar` and `etc` from `program`,
and assign each uniform to the corresponding variable.
This macro transform the following expression:
```julia
@uniforms foo, bar, qux: baz = program
```
Into:
```julia
foo = getuniform(program, :foo)
bar = getuniform(program, :bar)
qux = getuniform(program, :baz)
```
You can also rename the variables as following:
```julia
@uniforms a: foo, b: bar = program
```
The variable `a` contains the location of `foo`,
and `b` contains the loctation of `bar`.
"""
macro uniforms(ex)
    @assert ex.head == :(=)
    program = esc(ex.args[2])
    if !Meta.isexpr(ex.args[1], :tuple)
        args = [ex.args[1]]
    else
        args = ex.args[1].args
    end
    return Expr(:block,
        map(args) do arg
            if arg isa Symbol
                alias, name = arg, arg
            elseif Meta.isexpr(arg, :(=))
                alias, name = arg.args
            elseif Meta.isexpr(arg, :call)
                alias, name = arg.args[2:3]
            end
            return Expr(:(=), esc(alias), Expr(:call, :getuniform, program, QuoteNode(name)))
        end...
    )
end

macro uniforms!(ex)
    args = ex.args
    for i in eachindex(args)
        if Meta.isexpr(args[i], :(:=))
            name, value = args[i].args
            args[i] = Expr(:call, :uniform!, name, value)
        end
    end
    return esc(Expr(:block, args...))
end















############### my original part




# """
# function cotrolling the window  for displaying CT scan  - min white and max max_shown_black
#     uniformsStore - instantiated object holding references to uniforms controlling displayed window
#  """
# function setCTWindow(min_shown_whiteInner::Int32, max_shown_blackInner::Int32, uniformsStore::MainImageUniforms)
#     @uniforms! begin
#         uniformsStore.min_shown_white := min_shown_whiteInner
#         uniformsStore.min_shown_black := max_shown_blackInner
#         uniformsStore.displayrange := Float32(min_shown_whiteInner - max_shown_blackInner)
#     end
# end

"""
sets color of the mask

"""
function setMaskColor(color::RGB, uniformsStore::MaskTextureUniforms)
    @uniforms! begin
        uniformsStore.colorsMaskRef := Cfloat[color.r, color.g, color.b, 0.8]
    end

end#setMaskColor

# """
# sets color of the mask

# """
# function isMaskDiffViss(isMaskDiffrenceVisUnifs)
#     @uniforms! begin
#         isMaskDiffrenceVisUnifs := 1
#     end

# end#isMaskDiffViss



"""
sets visibility of the texture
"""
function setTextureVisibility(isvisible::Bool, uniformsStore::TextureUniforms)
    @uniforms! begin
        uniformsStore.isVisibleRef := isvisible ? 1 : 0
    end

end#setTextureVisibility


"""
sets minimum and maximum value for display -
    in case of continuus colors it will clamp values - so all above max will be equaled to max ; and min if smallert than min
    in case of main CT mask - it will controll min shown white and max shown black
    in case of maks with single color associated we will step data so if data is outside the rande it will return 0 - so will not affect display
"""
function coontrolMinMaxUniformVals(textur::TextureSpec)
    newMin = textur.minAndMaxValue[1]
    newMax = textur.minAndMaxValue[2]
    uniformsStore = textur.uniforms
    range = newMax - newMin
    if (range < 1)
        range = 1
    end#if

    @uniforms! begin
        uniformsStore.maskMinValue := newMin
        uniformsStore.maskMAxValue := newMax
        uniformsStore.maskRangeValue := range
    end

end#coontrolMinMaxUniformVals

"""
controlls contribution  of given mask to the overall image - maximum value is 1 minimum 0 if we have 3 masks and all control contribution is set to 1 and all are visible their corresponding influence to pixel color is 33%
      if plus is pressed it will increse contribution by 0.1
      if minus is pressed it will decrease contribution by 0.1
it also modifies given TextureSpec
change - how should the texture spec be modified
"""
function changeTextureContribution(textur::TextureSpec, change::Float32)
    newValue = textur.maskContribution + change
    if (newValue >= 0 && newValue <= 1)
        textur.maskContribution = newValue
        @uniforms! begin
            textur.uniforms.maskContribution := newValue

        end#@uniforms!
    end#if

end#changeTextureContribution

function changeMainTextureContribution(textur::TextureSpec, change::Float32, stateObject::StateDataFields)
    newValue = textur.maskContribution + change
    if (newValue >= 0 && newValue <= 1)
        textur.maskContribution = newValue
        @uniforms! begin
            actor.actor.mainForDisplayObjects.mainImageUniforms.mainImageContribution := newValue
        end#@uniforms!
    end#if

end#changeTextureContribution









end #module




