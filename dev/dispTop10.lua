--------------------------------------------------------------------------------
-- Display the top10 test images
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'image'
data = torch.load('Top10TestData.t7')

for label, cls in pairs(data) do
   image.display{image = cls.image, legend = label, nrow = 5}
end
