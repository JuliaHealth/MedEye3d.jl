

getOrCreateMaskData(Int64, "liver", "trainingScans/liver-orig001", (10,10,10), RGBA(0,0,255,0.4))






exmpleH = @spawnat persistenceWorker Main.h5manag.getExample()
arrr= fetch(exmpleH)

minimumm = -1000

maximumm = 2000
imageDim = size(arrr)
maskF = @spawnat persistenceWorker Main.h5manag.getOrCreateMaskData(Int16, "liverOrganMask", "trainingScans/liver-orig001", imageDim, RGBA(0,0,255,0.4))
mask = fetch(maskF)

using Main.imageViewerHelper
using Main.MyImgeViewer


singleCtScanDisplay(arrr, [mask],minimumm, maximumm)


widerInd = unique ∘ collect ∘ Iterators.flatten ∘ map((c->Main.imageViewerHelper.cartesianCoordAroundPoint(c,2)),Main.imageViewerHelper.cartesianCoordAroundPoint(CartesianIndex(0,0,0),2) ) 

arra = mask.maskArrayObs[]


#this will find all the cartesian indexes related to place clicked
#this will find all the cartesian indexes related to pixels around clicked patch
#we connect those indexes
collectedIndexes = filter((ind) -> arra[ind] == 0.5, CartesianIndices(arra))
# now we need to separate patches we will do it by first sorting by sth like k means clustering so we will store in a list all of the indexes that are no more than 2 units from given index
#we use sets to increase speed of finding intersections  also sets are naturally unique  so we will not need additional operations
closeIndexes = map(((ind) ->Set(
                    filter( (innerind) -> 
                    abs(imageViewerHelper.cartesianTolinear(innerind)  - imageViewerHelper.cartesianTolinear(ind))<3
                         ,collectedIndexes)))
                    ,collectedIndexes)
# next we will concatenate all sets that have any common element
```@doc
we are checking weather any set in arr have some common element with setA if yes wefuse them if not we add setA to arr
setA- first set to check
arr - array of resulting sets
  ```
function ifIntersectAdd(arr, setA) 
if any intersect
end #ifIntersectAdd


closeIndexes = 
reduce(+, [], [1,2,3,4,5])  


intersect
# after we concateneded thos we will take only unique elements







# @spawnat persistenceWorker Main.h5manag.saveMaskData!(Int16, mask)

zz = @spawnat persistenceWorker Main.h5manag.saveMaskData!(Int16, mask)
fetch(zz)


exmpleH = getExample()
imageDim = size(exmpleH)
mask =getOrCreateMaskData(Int16, "liverOrganMask", "trainingScans/liver-orig001", imageDim, RGBA(0,0,255,0.4))
saveMaskDataC!(Int16, mask)


