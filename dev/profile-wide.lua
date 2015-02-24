--------------------------------------------------------------------------------
-- Profiling bigger-than-eye algorithm
--------------------------------------------------------------------------------
-- Alfredo Canziani, Feb 15
--------------------------------------------------------------------------------

require 'image'
require 'getSpatialModel'

-- Profiling -------------------------------------------------------------------

dash = '--------------------------------------------------------------------------------\n'

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- Eye-size, no SoftMax(), spatial
print(dash..'Profiling eye-size, no SoftMax(), spatial\n')
img = top10.phone.image[1]
count = 1000
collectgarbage()
timer = torch.Timer()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
end
cutorch.synchronize()
forwardTime = timer:time().real / count
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   gradInput = model:updateGradInput(img:cuda(), gradLoss)
end
cutorch.synchronize()
forwardAndGradTime = timer:time().real / count
print('Forward time          / sample [ms]: ', 1e3 * forwardTime)
print('Forward and grad time / sample [ms]: ', 1e3 * forwardAndGradTime)
print('Just grad time        / sample [ms]: ', 1e3 * (forwardAndGradTime - forwardTime))

-- 1067x1600, no SoftMax(), spatial
print(dash..'Profiling 1067x1600, no SoftMax(), spatial\n')
img = image.load('../imgs/peep01.jpg')
count = 100
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
end
cutorch.synchronize()
forwardTime = timer:time().real / count
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   gradInput = model:updateGradInput(img:cuda(), gradLoss)
end
cutorch.synchronize()
forwardAndGradTime = timer:time().real / count
print('Forward time          / sample [ms]: ', 1e3 * forwardTime)
print('Forward and grad time / sample [ms]: ', 1e3 * forwardAndGradTime)
print('Just grad time        / sample [ms]: ', 1e3 * (forwardAndGradTime - forwardTime))

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- Eye-size, spatial, bottom-up
print(dash..'Profiling eye-size, spatial, bottom-up\n')
img = top10.phone.image[1]
model:forward(img:cuda())
psProb = torch.CudaTensor(#model.output)
count = 1000
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end
end
cutorch.synchronize()
forwardTime = timer:time().real / count
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end
   gradInput = model:updateGradInput(img:cuda(), psProb) -- bottom-up
end
cutorch.synchronize()
forwardAndGradTime = timer:time().real / count
print('Forward time          / sample [ms]: ', 1e3 * forwardTime)
print('Forward and grad time / sample [ms]: ', 1e3 * forwardAndGradTime)
print('Just grad time        / sample [ms]: ', 1e3 * (forwardAndGradTime - forwardTime))

-- 1067x1600, spatial, bottom-up
print(dash..'Profiling 1067x1600, spatial, bottom-up\n')
img = image.load('../imgs/peep01.jpg')
model:forward(img:cuda())
psProb = torch.CudaTensor(#model.output)
count = 100
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end
end
cutorch.synchronize()
forwardTime = timer:time().real / count
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end
   gradInput = model:updateGradInput(img:cuda(), psProb) -- bottom-up
end
cutorch.synchronize()
forwardAndGradTime = timer:time().real / count
print('forward time          / sample [ms]: ', 1e3 * forwardTime)
print('forward and grad time / sample [ms]: ', 1e3 * forwardAndGradTime)
print('just grad time        / sample [ms]: ', 1e3 * (forwardAndGradTime - forwardTime))

-- 1067x1600, spatial, bottom-up, SpatialSoftMax
print(dash..'Profiling 1067x1600, spatial, bottom-up, SpatialSoftMax\n')
img = image.load('../imgs/peep01.jpg')
model:forward(img:cuda())
count = 100
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   psProb = nn.SpatialSoftMax(model.output)
end
cutorch.synchronize()
forwardTime = timer:time().real / count
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   psProb = nn.SpatialSoftMax(model.output)
   gradInput = model:updateGradInput(img:cuda(), psProb) -- bottom-up
end
cutorch.synchronize()
forwardAndGradTime = timer:time().real / count
print('Forward time          / sample [ms]: ', 1e3 * forwardTime)
print('Forward and grad time / sample [ms]: ', 1e3 * forwardAndGradTime)
print('Just grad time        / sample [ms]: ', 1e3 * (forwardAndGradTime - forwardTime))

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--[[
-- Eye-size, spatial, top-down
print('profiling eye-size, spatial, top-down')
img = top10.phone.image[1]
model:forward(img:cuda())
psProb = torch.CudaTensor(#model.output)
target = psProb:clone():zero()
count = 1000
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end
end
cutorch.synchronize()
forwardTime = timer:time().real / count
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end
   target[2] = psProb[2]:clone():mul(-1) -- top-down
   gradInput = model:updateGradInput(img:cuda(), target)
end
cutorch.synchronize()
forwardAndGradTime = timer:time().real / count
print('Forward time          / sample [ms]: ', 1e3 * forwardTime)
print('Forward and grad time / sample [ms]: ', 1e3 * forwardAndGradTime)
print('Just grad time        / sample [ms]: ', 1e3 * (forwardAndGradTime - forwardTime))

-- 1067x1600, spatial, top-down
print('profiling 1067x1600, spatial, top-down')
img = image.load('../imgs/peep01.jpg')
model:forward(img:cuda())
psProb = torch.CudaTensor(#model.output)
target = psProb:clone():zero()
count = 100
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end
end
cutorch.synchronize()
forwardTime = timer:time().real / count
collectgarbage()
timer:reset()
for i = 1, count do
   gradLoss = model:forward(img:cuda()) -- if getspatialmodel
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end
   target[2] = psProb[2]:clone():mul(-1) -- top-down
   gradInput = model:updateGradInput(img:cuda(), target)
end
cutorch.synchronize()
forwardAndGradTime = timer:time().real / count
print('Forward time          / sample [ms]: ', 1e3 * forwardTime)
print('Forward and grad time / sample [ms]: ', 1e3 * forwardAndGradTime)
print('Just grad time        / sample [ms]: ', 1e3 * (forwardAndGradTime - forwardTime))

--]]
