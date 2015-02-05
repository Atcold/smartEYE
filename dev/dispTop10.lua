--------------------------------------------------------------------------------
-- Display (or save) the top10 test images
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'image'
data = torch.load('../data/17cate9filter/Top10TestData.t7')

for label, cls in pairs(data) do
   -- Display
   image.display{image = cls.image, legend = label, nrow = 5}

   -- Save
   -- image.saveJPG(label .. '.jpg',image.toDisplayTensor{
   --    input = cls.image, legend = label, nrow = 5
   -- })
end
