--------------------------------------------------------------------------------
-- Generate publication "figure 2"
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14, Jan/Feb 15
--------------------------------------------------------------------------------

require 'image'
require 'getSpatialModel'

-- Function definition ---------------------------------------------------------
image.fit = function (inputImage)
   return image.scale(inputImage:float(), img:size(3)/4, img:size(2)/4, 'simple')
end

image.scaleDown = function (inputImage)
   return image.scale(inputImage:float(), img:size(3)/4, img:size(2)/4)
end

clip = function (input)
   input[input:gt(input:max()/4)] = input:max()/4
   input[input:lt(input:min()/4)] = input:min()/4
   return input
end

gradManipulation = function (grad)
   out = grad:float():abs():max(1):repeatTensor(3,1,1)
   return clip(out)
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
person = 2
target = psProb:clone():zero()
target[person] = psProb[person]:clone():mul(-1)
gradInput = model:updateGradInput(img:cuda(), target)

-- Visulaisation
--image.display{image = gradInput:float(), zoom = 0.4,
pub2 = image.toDisplayTensor{ padding = 2,
   nrow = 2, scaleeach = true,
   legend = '"person" pseudo probability and top-down saliency map',
   input = {
--   image = {
      image.scaleDown(img:float()),
      image.fit(image.y2jet(psProb[2]*255+1)),
      image.scaleDown(clip(gradInput:float())),
      image.scaleDown(gradManipulation(gradInput)),
   }
}
image.save('pub2.png',pub2)
