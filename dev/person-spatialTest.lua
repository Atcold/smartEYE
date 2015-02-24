--------------------------------------------------------------------------------
-- Generate several spatial saliency map for a person test image
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14, Jan/Feb 15
--------------------------------------------------------------------------------

require 'image'
require 'getSpatialModel'

-- Function definition ---------------------------------------------------------
image.fit = function (inputImage)
   return image.scale(inputImage:float(), img:size(3), img:size(2), 'simple')
end
gradManipulation = function (grad, th)
   return grad:float():abs():max(1):repeatTensor(3,1,1)
end

-- Main program ----------------------------------------------------------------
img = image.load('../imgs/peep01.jpg')
model:forward(img:cuda())

-- Estimating pseudo-probability, this has to be rewritten with FFI
psProb = torch.CudaTensor(#model.output)
for i = 1, model.output:size(2) do
   for j = 1, model.output:size(3) do
      psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
   end
end

-- Top-down saliency
target = psProb:clone():zero()
target[2] = psProb[2]:clone():mul(-1)
gradInput = model:updateGradInput(img:cuda(), target)

-- Visulaisation
image.display{image = gradInput:float(), zoom = 0.4,
--pub2 = image.toDisplayTensor{ padding = 8,
   nrow = 2, scaleeach = true,
   legend = '"person" pseudo probability and top-down saliency map',
--   input = {
   image = {
      img:float(),
      image.fit(image.y2jet(psProb[2]*255+1)),
      gradInput:float(),
      gradManipulation(gradInput, 0)
   }
}
--image.save('pub2.png',image.scale(pub2,pub2:size(3)/2,pub2:size(2)/2))
