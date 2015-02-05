--------------------------------------------------------------------------------
-- Generate spatial prediction map
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'image'
require 'gnuplot'
require 'imgraph'
require 'getSpatialModel'

img = image.load('../imgs/peep01.jpg')
model:forward(img:cuda())
--gnuplot.hist(model.output[revClas.jacket])
psProb = torch.CudaTensor(#model.output)

-- This has to be rewritten with FFI
for i = 1, model.output:size(2) do
   for j = 1, model.output:size(3) do
      psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
   end
end

--gnuplot.hist(rescaled[revClas.jacket])

-- image.display{
--    image = rescaled[revClas.jacket]:float(),
--    zoom = 15,
--    legend = 'Spatial person with jaket'
-- }

colorMap = imgraph.colorize(psProb[2]:float()*255,image.jetColormap(256):float())
image.display{zoom = 0.4, nrow = 2, image = {
   img,
   image.scale(colorMap, img:size(3), img:size(2)),
   image.scale(colorMap, img:size(3), img:size(2), 'simple'),
   img*0
}}
