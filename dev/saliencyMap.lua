--------------------------------------------------------------------------------
-- Generate saliency map
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'image'
require 'getModel'
--require 'getSpatialModel' -- remove preprocessing for fairness

io.write('[V]iew or [P]rofile? (V/P) ')
userInput = io.read()
if userInput == 'V' or userInput == 'v' then
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
else
   print('Profiling...')
   count = 0
   timer = torch.Timer()
   for i = 1, 10 do
      for label, class in pairs(top10) do
         for n, img in ipairs(class.image) do
            gradLoss = loss:updateGradInput(model:forward(img:cuda()), revClas[label])
            --gradLoss = model:forward(img:cuda()) -- if getSpatialModel
            count = count + 1
         end
      end
   end
   cutorch.synchronize()
   forwardTime = timer:time().real / count
   timer:reset()
   for i = 1, 10 do
      for label, class in pairs(top10) do
         for n, img in ipairs(class.image) do
            gradLoss = loss:updateGradInput(model:forward(img:cuda()), revClas[label])
            --gradLoss = model:forward(img:cuda()) -- if getSpatialModel
            gradInput = model:updateGradInput(img:cuda(), gradLoss)
         end
      end
   end
   cutorch.synchronize()
   forwardAndGradTime = timer:time().real / count
   print('Forward time          / sample [ms]: ', 1e3 * forwardTime)
   print('Forward and grad time / sample [ms]: ', 1e3 * forwardAndGradTime)
   print('Just grad time        / sample [ms]: ', 1e3 * (forwardAndGradTime - forwardTime))
end
