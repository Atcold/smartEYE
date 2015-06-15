--------------------------------------------------------------------------------
-- Generate publication "figure 1"
--------------------------------------------------------------------------------
-- Alfredo Canziani, Feb 15
--------------------------------------------------------------------------------

require 'image'
require 'getModel'
require 'ext'

-- Function definition ---------------------------------------------------------
clip = function (input)
   input[input:gt(input:max()/3)] = input:max()/3
   input[input:lt(input:min()/3)] = input:min()/3
   return input
end

-- Main program ----------------------------------------------------------------
nbs = {4, 10, 1, 6}
label = {'phone', 'plant', 'laptop', 'laptop'}
img = {}
gradInput = {}

for i, n in ipairs(nbs) do
   img[i] = top10[label[i]].image[n]
   gradLoss = loss:updateGradInput(model:forward(img[i]:cuda()), revClas[label[i]])
   gradInput[i] = model:updateGradInput(img[i]:cuda(), gradLoss):float()
end

--image.display{image = {
pub1 = image.toDisplayTensor{input = {
      img[1], clip(gradInput[1]:clone():abs():max(1):repeatTensor(3,1,1)),
      img[2], clip(gradInput[2]:clone():abs():max(1):repeatTensor(3,1,1)),
      img[3], clip(gradInput[3]:clone():abs():max(1):repeatTensor(3,1,1)),
      img[4], clip(gradInput[4]:clone():abs():max(1):repeatTensor(3,1,1)),
   }, scaleeach = true, nrow = 4, zoom = 2, padding = 2
}
image.save('pub1.png',pub1)
