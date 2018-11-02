function [ img , pts , box ] = cropImage( img , pts )


deltx = round( max( pts( :,1 ) ) - min( pts( :,1 ) ) );
delty = round( max( pts( :,2 ) ) - min( pts( :,2 ) ));

leftx = round( max( min( pts( :,1 ) )-deltx/4, 1 ) );
rightx =round(  min( max( pts( :,1 ) )+deltx/4, size( img,2 ) ));

lefty = round( max( min( pts( :,2 ) )-delty, 1 ) );
righty = round( min( max( pts( :,2 ) )+1, size( img,1 ) ) );

img = img( lefty:righty,leftx:rightx,: );
pts(:,1) = pts(:,1) - leftx + 1;
pts(:,2) = pts(:,2) - lefty + 1;
box = [leftx lefty] - 1;
% pts(:,1) = pts(:,1) - leftx + 1;
% pts(:,2) = pts(:,2) - lefty + 1 ;
% box = [leftx lefty]  ;