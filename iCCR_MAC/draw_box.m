function [ projBox , loc , rot ] = draw_box( pts , PDM )

[ ~ , gparams ] = convert2D_to_3D( pts , PDM );


box_verts = [-1, 1, -1,;
			1, 1, -1;
		    1, 1, 1;
            -1, 1, 1;
			1, -1, 1;
			1, -1, -1;
			-1, -1, -1;
			-1, -1, 1];
        
pitch = gparams(2);
yaw = gparams(3);
roll = gparams(4);
roll = -roll;
rotmat = Euler2RotRaw( pitch, yaw, roll, true );

box = box_verts * 100;


tx = gparams(5);
ty = gparams(6);
tz = gparams(7);
rotBox = box * rotmat;
rotBox(:,1) = rotBox(:,1) + tx;
rotBox(:,2) = rotBox(:,2) + ty;
rotBox(:,3) = rotBox(:,3) + tz;
projBox = rotBox;


        %pts = CalcShape2D( PDM , plocal, pglobl(1:6) ); pts = reshape(pts(1:132),[],2);
        loc = mean( pts );
        rot = Euler2Rot( gparams(1:6) );

end

function [ pts3D , pglobl , local  ] = convert2D_to_3D( pts2D  , PDM )


if nargin < 2
load('CLM_Shape_Model.mat','PDM');
end
pts3D = zeros(66,3,size(pts2D,3));

for i = 1 : size( pts2D , 3 )

     [ local , pglobl ] = CalcParams( PDM , reshape(pts2D(:,:,i),[],1) );
     pts3D(:,:,i) = reshape(CalcShape2D( PDM, local, pglobl ),66,3);
      
    
end




end
function s = CalcShape2D(obj,plocal,pglobl)

    n = size(obj.M,1)/3;
    a = pglobl(1,1);
    x = pglobl(5,1);
    y = pglobl(6,1);
    z = pglobl(7,1);
    R_ = Euler2Rot(pglobl(1:6));
    S_ = obj.M + obj.V*plocal;
    s = zeros(2*n,1);
    
    for i = 1:n
    
        s(i,1) = a*( R_(1,1)*S_(i,1) + R_(1,2)*S_(i+n,1) + R_(1,3)*S_(i+n*2,1) ) + x;
        s(i+n,1) = a*( R_(2,1)*S_(i,1) + R_(2,2)*S_(i+n,1) + R_(2,3)*S_(i+n*2,1) ) + y;
         s(i+2*n,1) = a*( R_(3,1)*S_(i,1) + R_(3,2)*S_(i+n,1) + R_(3,3)*S_(i+n*2,1) ) + z;

    end 
    
end

function RResult = Euler2Rot(p,full)
    switch nargin
        case 1
            full = true;
        case 2
            
        otherwise
            error('Lack of input arguments!');
    end
    
    if size(p,1) ~= 6 || size(p,2) ~= 1
        error('Matrix dimension mismatched!');
    end
    
    RResult = Euler2RotRaw(p(2,1),p(3,1),p(4,1),full);
end



function [local,pglobl] = CalcParams(obj,s)
    
    n = size(obj.M,1)/3;    
    p = zeros(size(obj.V,2),1);
    pglobl = zeros(7,1);
    local = zeros(size(obj.V,2),1);
    
    for iter = 1:100
        
        tempShape = CalcShape3D(obj,local);
        [scale,pitch,yaw,roll,tx,ty,tz] = Align3Dto2DShapes(s,tempShape);                
        
        R = Euler2RotRaw(pitch,yaw,roll);

        z = scale*reshape(tempShape,[n,3])*(R(3,:)');
        si = 1/scale;

        Tx = -si*(R(1,1)*tx + R(2,1)*ty);
        Ty = -si*(R(1,2)*tx + R(2,2)*ty);
        Tz = -si*(R(1,3)*tx + R(2,3)*ty);
        
        tempShape = [s(1:n) s(n+1:2*n) z]*R;                
        tempShape(:,1) = si*tempShape(:,1) + Tx;
        tempShape(:,2) = si*tempShape(:,2) + Ty;
        tempShape(:,3) = si*tempShape(:,3) + Tz;                       

        local = obj.V' * (tempShape(:) - obj.M);

        if norm(local-p,'fro') < 1.0e-5        
            break;        
        end
        
        p = local;
        
    end
     
    pglobl(1) = scale;
    pglobl(2) = pitch;
    pglobl(3) = yaw;   
    pglobl(4) = roll;
    pglobl(5) = tx;    
    pglobl(6) = ty;   
    pglobl(7) = tz;
    
end


function sResult = CalcShape3D(obj,plocal)
    
    sResult = obj.M + obj.V*plocal;
    
end


function [scale,pitch,yaw,roll,x,y,z] = Align3Dto2DShapes(s2D_cpy,s3D_cpy)
    
    if size(s2D_cpy,2) ~= 1 || size(s3D_cpy,1) ~= 3*(size(s2D_cpy,1)/2) ...
            || size(s3D_cpy,2) ~= 1
        error('Matrix dimension mismatched!');
    end
    
    n = size(s2D_cpy,1)/2;    
    
    rowsX = size(s2D_cpy,1)/2;    
    X = reshape(s2D_cpy,[rowsX,2]);
    
    rowsS = size(s3D_cpy,1)/3;
    S = reshape(s3D_cpy,[rowsS,3]);    
    
    t2 = sum(X)/n;        
    for i = 1:2        
        X(:,i) = X(:,i) - t2(i);
    end        
    t3 = sum(S)/n;
    for i = 1:3        
        S(:,i) = S(:,i) - t3(i);
    end
    
    %---------Jason's code used cholesky inverse----------%
    M = (S'*S)\(S'*X);
    
    MtM = M'*M;
    [u,w,vt] = svd(MtM,0);
    w = diag(w);
    w(1,1) = 1/sqrt(w(1,1));
    w(2,1) = 1/sqrt(w(2,1));
    
    T = zeros(3,3);
    T(1:2,1:3) = u*diag(w)*vt*M';
    % T(1:2,1:3) = inv( M'*M )^0.5 * M';
    
    scale = 0.5*sum(sum( T(1:2,1:3).*(M') ));
    T = AddOrthRow(T);
    [pitch,yaw,roll] = Rot2Euler(T);
    T = T*scale;
    
    x = t2(1) - ( T(1,1)*t3(1) + T(1,2)*t3(2) + T(1,3)*t3(3) );
    y = t2(2) - ( T(2,1)*t3(1) + T(2,2)*t3(2) + T(2,3)*t3(3) );
    z = 0 - ( T(3,1)*t3(1) + T(3,2)*t3(2) + T(3,3)*t3(3) );
    
end


function RResult = AddOrthRow(R)
    if size(R,1) ~= 3 || size(R,2) ~= 3
        error('Matrix dimension mismatched!');
    end
    
    R(3,1) = R(1,2)*R(2,3) - R(1,3)*R(2,2);
    R(3,2) = R(1,3)*R(2,1) - R(1,1)*R(2,3);
    R(3,3) = R(1,1)*R(2,2) - R(1,2)*R(2,1);
    RResult = R;    
end


function [pitch,yaw,roll] = Rot2Euler(R)
    if size(R,1) ~= 3 || size(R,2) ~= 3
        error('Matrix dimension mismatched!');
    end        
    
    q = zeros(4,1);
    q(1) = sqrt( 1+R(1,1)+R(2,2)+R(3,3) ) / 2;
    q(2) = ( R(3,2)-R(2,3) ) / ( 4*q(1)+eps );
    q(3) = ( R(1,3)-R(3,1) ) / ( 4*q(1)+eps );
    q(4) = ( R(2,1)-R(1,2) ) / ( 4*q(1)+eps );
    
    yaw = asin( 2*(q(1)*q(3)+q(2)*q(4)) );
    
    pitch = atan2( 2*(q(1)*q(2)-q(3)*q(4)), q(1)*q(1) - q(2)*q(2) - q(3)*q(3) + q(4)*q(4) );
    
    roll = atan2( 2*(q(1)*q(4)-q(2)*q(3)), q(1)*q(1) + q(2)*q(2) - q(3)*q(3) - q(4)*q(4) );
    
end

function RResult = Euler2RotRaw(pitch,yaw,roll,full)

    switch nargin
        case 3
            full = true;
        case 4
            
        otherwise
            error('Lack of input arguments!');
    end
    
    if full
        
        R = zeros(3,3);
        
    else
        
        R = zeros(2,3);
        
    end
    
    sina = sin(pitch);  sinb = sin(yaw); sinc = sin(roll);
    cosa = cos(pitch);  cosb = cos(yaw); cosc = cos(roll);
    
    R(1,1) = cosb*cosc; 
    R(1,2) = -cosb*sinc; 
    R(1,3) = sinb;
    R(2,1) = cosa*sinc + sina*sinb*cosc;
    R(2,2) = cosa*cosc - sina*sinb*sinc;
    R(2,3) = -sina*cosb;
    
    if full
        R = AddOrthRow(R);
    end
    
    RResult = R;
end
