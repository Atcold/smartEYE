--------------------------------------------------------------------------------
-- Generate saliency map
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'image'
require 'getModel'

for label, class in pairs(top10) do
   io.write('Processing ' .. label); io.flush()
   for n, img in ipairs(class.image) do
      gradLoss = loss:updateGradInput(model:forward(img:cuda()), revClas[label])
      gradInput = model:updateGradInput(img:cuda(), gradLoss)
      image.display{
         image = {
            img,
            gradInput,
            gradInput:clone():max(1):repeatTensor(3,1,1),
            gradInput:clone():abs():max(1):repeatTensor(3,1,1)
         }, scaleeach = true, nrow = 2, zoom = 2, legend = label..'['..n..']'
      }
   end
   io.write(". Press enter for next category, 'q' for quitting... ")
   if io.read() == 'q' then break end
end
