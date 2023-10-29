"""
supported 'dataType' attributes for dataSets
"CT" - will lead to classical display of CT image
"boolLabel" - binary color display (either some color or none)
"multiDiscreteLabel" - diffrent discrete colors assigned to multiple discrete labels
"contLabel" - continous color  display for example float valued probabilistic label
"PET" - continous color  display for example float valued 
"manualModif" - manually modified array - using annotation functionality of a viewer

"""

module visualizationFromHdf5
  

export getGroupOrCreate,getArrByName,loadFromHdf5Prim,getSomeColor,openHDF5,calculateAndDisplay,writeGroupAttribute,refresh
import ..StructsManag
import ..StructsManag.threeToTwoDimm

import ..ForDisplayStructs
import ..ForDisplayStructs.TextureSpec
using ColorTypes
import ..SegmentationDisplay
using HDF5
import ..DataStructs.ThreeDimRawDat
import ..DataStructs.DataToScrollDims
import ..DataStructs.FullScrollableDat
import ..ForDisplayStructs.KeyboardStruct
import ..ForDisplayStructs.MouseStruct
import ..ForDisplayStructs.ActorWithOpenGlObjects
import ..OpenGLDisplayUtils
import ..DisplayWords.textLinesFromStrings
import ..StructsManag.getThreeDims
using ..DisplayWords
using ..distinctColorsSaved
using  ..OpenGLDisplayUtils,  ..ForDisplayStructs
using   ..Uniforms,   ..CustomFragShad,  ..DataStructs,  ..DisplayWords
using   ..OpenGLDisplayUtils,  ..ForDisplayStructs,..TextureManag
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using ..SegmentationDisplay

"""
simply open dataset
"""
function openHDF5(pathToHDF5)
  return  h5open(pathToHDF5, "r+")

end

"""
Writes group attribute - simple wrapper
group - object representing the group
attrName - name of the attribute
value - value of the attribute
"""
function writeGroupAttribute(fid,groupName, attrName, value)
  group = fid[groupName]
  write_attribute(group, attrName, value)
end


"""
return the HDF5 group if it is not already present in dataset it creates it 
  fid - HDF5 object 
  groupName - string representing group of intrest 
"""
function getGroupOrCreate(fid, groupName)
  if(!haskey(fid, groupName))
    return create_group(fid, groupName)
  end
  #in case it is already created
  return fid[groupName]
    
end


"""
get some color from listOfColors or if those are already used some random color from 
listOfColorUsed- boolean array marking which colors were already used from listOfColors
"""
function getSomeColor(listOfColorUsed)
  if(sum(listOfColorUsed)<18)
  for i in 1:18
    if(!listOfColorUsed[i])
      listOfColorUsed[i]=true; 
      tupl= listOfColors[i]
      return RGB(tupl[1]/255,tupl[2]/255,tupl[3]/255)
    end#if
  end#for
else 
   #if we are here it means that no more colors from listOfColors is available - so we need to take some random color from bigger list
   tupl = longColorList[rand(Int,1:255)]

   return  RGB(tupl[1]/255,tupl[2]/255,tupl[3]/255)
end
end#getSomeColor

"""
becouse Julia arrays is column wise contiguus in memory and open GL expects row wise we need to rotate and flip images 
pixels - 3 dimensional array of pixel data 
"""
function permuteAndReverse(pixels)
    
    pixels=  permutedims(pixels, (3,2,1))
    sizz=size(pixels)
    for i in 1:sizz[2]
        for j in 1:sizz[3]
            pixels[:,i,j] =  reverse(pixels[:,i,j])
        end# 
    end# 
    return pixels
  end#permuteAndReverse

function onlyPermute(pixels)    
    pixels=  permutedims(pixels, (3,2,1))
    
    sizz=size(pixels)
    for i in 1:sizz[1]
        for j in 1:sizz[3]
            pixels[i,:,j] =  reverse(pixels[i,:,j])
        end# 
    end#

    return pixels
  end#permuteAndReverse
  

"""
loading image from HDF5 dataset 
fid- objct managing HDF5 file
patienGroupName- tell us  the string that is a name of the HDF5 group with all data needed for a given patient
fractionOfMainIm - effectively will set how much space is left for text display
addTextSpecs - additional texture specifications required to define and save some of the masks
"""
function loadFromHdf5Prim(fid, patienGroupName::String
  , addTextSpecs::Vector{TextureSpec},listOfColorUsed,fractionOfMainIm::Float32= Float32(0.8) )

#marks what colors are already used 
  group = fid[patienGroupName]
 #strings holding the arrays holding data about given patient
 imagesMasks = keys(group)
print("imagesMasks $(imagesMasks) ")

#adding one spot to be able to get manually modifiable mask
 imageSize::Tuple{Int64,Int64,Int64}= (0,0,0)
toAddAddedTextures = 0
addTextureNames::Vector{String}= Vector(undef,length(addTextSpecs))
#making sure that each array will be included
index = 0;
for tex in addTextSpecs
  index+=1
  addTextureNames[index]=tex.name
  if(!haskey(group,tex.name))
    toAddAddedTextures+=1
  end#if  
end#for

#preinitialized placeholders
 textureSpecifications::Vector{TextureSpec}=Vector(undef,length(imagesMasks)+toAddAddedTextures)



 tupleVect=Vector(undef,length(imagesMasks)+toAddAddedTextures)

 index = 0;
  for maskName in imagesMasks
    index+=1
    dset = group[maskName]
    dataTypeStr= attributes(dset)["dataType"][]
    #additional arrays are saved and loaded as is
    voxels =  dset[:,:,:]


    # @info " mask name   " maskName
    # @info "voxel dims   " size(voxels)

    typp = eltype(voxels)
    min = attributes(dset)["min"][]
    max = attributes(dset)["max"][]

#initializing texture
    textureSpec =  if(maskName in addTextureNames )
      textureSpec = filter(tex->tex.name==maskName,addTextSpecs  )[1]
    else getDefaultTextureSpec(dataTypeStr,maskName ,index,listOfColorUsed, typp, min, max,voxels)
  end  
  textureSpec.numb= index
    imageSize= size(voxels)

    textureSpecifications[index]=textureSpec
    #append!( textureSpecifications, textureSpec )
    tupleVect[index] =(maskName,voxels)
  end #for
#and additionally manually modifiable ...

for tex in addTextSpecs
  if( !haskey(group, tex.name) )
    index+=1

    #@info "nnnnnn no  manualModif key "
    textureSpec = tex
    textureSpec.numb= index
    textureSpecifications[index]=textureSpec
    tupleVect[index] =(tex.name,zeros(eltype(tex.minAndMaxValue) ,imageSize))
  end#if
end#for



print(textureSpecifications) #TODO (remove)

spacingList = attributes(group)["spacing"][]
spacing=(Float32(spacingList[1]),Float32(spacingList[2]),Float32(spacingList[3]))


datToScrollDimsB= ForDisplayStructs.DataToScrollDims(
    imageSize=  imageSize
    ,voxelSize=spacing
    ,dimensionToScroll = 3 );

slicesDat= getThreeDims(tupleVect )


#mainLines= textLinesFromStrings([""]);
#supplLines=[];

mainLines= textLinesFromStrings([" "]);
supplLines=map(x->  textLinesFromStrings(["slice $(x)"]), 1:imageSize[3] );



mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                 ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
                                 ,dataToScroll= slicesDat
                                 ,mainTextToDisp= mainLines
                                 ,sliceTextToDisp=supplLines );

                                 coordinateDisplay(textureSpecifications
                                 ,fractionOfMainIm ,datToScrollDimsB ,1000);


                                 passDataForScrolling(mainScrollDat);

return mainScrollDat

 end #loadFromHdf5Prim

 """
 given datatype string will return appropriate default configuration for image display
 """
function getDefaultTextureSpec(dataTypeStr::String,maskName::String ,index::Int,listOfColorUsed
      , typp, min, max,voxels)::TextureSpec

  print(" mmmmmmmmmmmm maskName $(maskName) min $(min) max $(max)  ")    
  if(dataTypeStr=="CT")
    return TextureSpec{typp}(
      name= maskName,
      isMainImage = true,
      minAndMaxValue= typp.([0,100])) 

  elseif(dataTypeStr=="boolLabel")
    return TextureSpec{typp}(
      name = maskName,
      color =getSomeColor(listOfColorUsed)
      ,minAndMaxValue= typp.([min,max])
     )
   
  elseif(dataTypeStr=="multiDiscreteLabel") 
    numberUniq=unique(voxels)
    colorSet=map(i->getSomeColor(listOfColorUsed),1: numberUniq )
    return TextureSpec{typp}(
      name = maskName,
      isContinuusMask=true,
      colorSet = colorSet
      ,minAndMaxValue= typp.([min,max])
     )
 
  elseif(dataTypeStr=="contLabel") 
    return TextureSpec{Float32}(
      name = maskName,
      isContinuusMask=true,
      colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
      ,minAndMaxValue= typp.([min,max])
     )

  elseif(dataTypeStr=="PET") 
    return TextureSpec{typp}(
      name = maskName,
      isContinuusMask=true,
      colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
      ,minAndMaxValue= typp.([min,max])
     )  


  end

end



function getArrByName(name::String,mainScrollDat )

return  filter((it)->it.name==name ,mainScrollDat.dataToScroll)[1].dat

end #getArrByName



function calculateAndDisplay(preparedDict,mainScrollDat::FullScrollableDat
  , conf::ConfigurtationStruct,numberToLookFor,goldArr,algoOutputGPU) 
  res= calcMetricGlobal(preparedDict,conf,goldArr,algoOutputGPU,numberToLookFor)
  append!(mainScrollDat.mainTextToDisp, textLinesFromStrings(giveStringsFromResultMetrics(res,conf)) )


end


  
  
"""
supplied ResultMetrics struct will return list of strings with results
"""
function giveStringsFromResultMetrics(res,conf::ConfigurtationStruct)::Vector{String}
  output = []
  if(conf.dice)
    append!(output,["dice $(res.dice)"])
  end  
  if(conf.jaccard)
    append!(output,["jaccard $(res.jaccard)"])
  end  
  if(conf.gce)
    append!(output,["gce $(res.gce)"])
  end  
  if(conf.vol)
    append!(output,["vol $(res.vol)"])
  end  
  if(conf.randInd)
    append!(output,["randInd $(res.randInd)"])
  end  
  if(conf.ic)
    append!(output,["ic $(res.ic)"])
  end  
  if(conf.kc)
    append!(output,["kc $(res.kc)"])
  end  
  if(conf.mi)
    append!(output,["mi $(res.mi)"])
  end  
  if(conf.vi)
    append!(output,["vi $(res.vi)"])
  end  
  if(conf.md)
    append!(output,["md $(res.md)"])
  end  
  if(conf.hd)
    append!(output,["hd $(res.hd)"])
  end  




  return output

end
  
"""
refresh image after some modifications are performed
actor - main actor processing GUI
"""
function refresh(actor)
  current = actor.actor.currentDisplayedSlice
  singleSlDat= actor.actor.onScrollData.dataToScroll|>
  (scrDat)-> map(threeDimDat->threeToTwoDimm(threeDimDat.type,Int64(current),actor.actor.onScrollData.dimensionToScroll,threeDimDat ),scrDat) |>
  (twoDimList)-> SingleSliceDat(listOfDataAndImageNames=twoDimList
                              ,sliceNumber=current
                              ,textToDisp = DisplayWords.getTextForCurrentSlice(actor.actor.onScrollData, Int32(current))  )

  updateImagesDisplayed(singleSlDat
                      ,actor.actor.mainForDisplayObjects
                      ,actor.actor.textDispObj
                      ,actor.actor.calcDimsStruct 
                      ,actor.actor.valueForMasToSet)

end#refresh
  
  


end#end



