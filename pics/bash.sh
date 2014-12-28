for i in `ls`
do
	if [ `ls $i | wc -l` == 1 ]
	then
		echo $i
		ls $i | wc -l
		rm -r $i	
	fi
done
