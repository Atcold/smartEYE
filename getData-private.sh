# Get data from server

path=${MYELAB?'Need to set MYELAB'}':~/elabshare/users/atcold/Git-data/smartEYE'

echo ''
echo 'Do you wish to download the network?'
select yn in 'Yes' 'No'; do
    case $yn in
        Yes ) scp -r $path'/net' .; break;;
        No ) break;;
    esac
done

echo ''
echo 'Do you wish to download sample images?'
select yn in 'Yes' 'No'; do
    case $yn in
        Yes ) scp -r $path'/imgs' .; break;;
        No ) break;;
    esac
done

echo ''
echo 'Do you wish to download top10 testing data?'
select yn in 'Yes' 'No'; do
    case $yn in
        Yes )
             mkdir -p data/17cate9filter
             scp -r $path'/data/17cate9filter/Top10TestData.t7' data/17cate9filter
             break;;
        No ) break;;
    esac
done

echo ''
echo 'Do you wish to download the WHOLE testing-set data?'
select yn in 'Yes' 'No'; do
    case $yn in
        Yes )
             mkdir -p data/17cate9filter
             scp -r $path'/data/17cate9filter/TestDataset.t7' data/17cate9filter
             break;;
        No ) break;;
    esac
done
