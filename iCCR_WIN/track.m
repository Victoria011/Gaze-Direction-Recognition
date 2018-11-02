function data = track( model , video , params  )

% - Enrique Sanchez-Lozano. University of Nottingham
% - Code released as is for research purposes only. 
% - Feel free to modify or redistribute.
% - Should you use the code, please cite
% - [1] E.Sanchez-Lozano, G. Tzimiropoulos, B. Martinez, F. De la Torre, M.
% Valstar. A Functional Regression approach to facial landmark tracking.
% IEEE TPAMI. 2017
% - [2] E.Sanchez-Lozano, B.Martinez, G.Tzimiropoulos, M.Valstar. Cascaded
% Continuous Regression for Real-time Incremental Face Tracking. ECCV 2016.
% - Feel free to contact me at Enrique.SanchezLozano@nottingham.ac.uk

% - the video string should be either [] for webcam, or the path to the
% video
% - params should contain the following components
% - params.show = 1 or 0 (show results or not)
% - params.incLearning = 1 or 0 (iCCR or simply CCR)
% - params.lambda = Incremental Learning update factor
% - params.error_fun = handle to function that will return whether the
% tracker is lost or not.
% - params.lost_model = the model that should be used by the error_fun

if nargin < 3
    % default params
    params.show = 1;
    params.incLearning = 1;
    params.lambda = 0.1 * [0.1 0.25 0.5 1];
end;


i = 1;

if ~isfield( params , 'error_fun' )
    params.error_fun = @detectLost_SVM;
    params.lost_model.SVM = model.SVM;
    params.lost_model.shModel = model.shModel;
end

if isempty(video)
    obj = webcam(1);
    obj.Resolution = '640x480';
    %obj.Resolution = '2304x1536';
else
    obj = VideoReader(video);
end;

global H %- the drawing scenario has been kindly adapted from the xx_track

%%% Initialise the data and the model given the input object
[ model , data ] = ccr_initialise( model , obj , video );

np = size( model.shModel.s0 , 1 );
incLearning = params.incLearning;
pts = [];

frameCount = 0;
keep = 1; first = 1;
while keep
    
    tic
    % - grab a frame
    if ~isempty(video)
        im = ( read( obj , i ) );     
    else
        im = snapshot( obj );
    end;
    if isfield(params,'transform')
    im = imrotate(im, params.transform.pm);
    im = imresize( im , params.transform.sc );
    if params.transform.flip
        im = im(:,end:-1:1,:);
    end
    end
   % im = im(:,end:-1:1,:);
    imcpy = im;
    
    % - track points
    % - usage: [ pts , img ] = ccr_track( im , pts , model );
    box = zeros(np,1);
    if isempty( pts )
        % - detect
        pts = detect_pts_SDM( im , model );
        
         %shape66( [ 61,65 ], :, : ) = [];
        if np == 68 && size( pts , 1 ) == 66
            pts = [pts(1:60,:); zeros(1,2); pts(61:63,:); zeros(1,2); pts(64:66,:)];
        end
        
        if ~isempty( pts )
            % - apply a tracking step
            [ imcrop , ptscrop , box ] = cropImage( im , pts );
            if size( imcrop , 1 ) * size( imcrop , 2 ) > 0
                im = imcrop; pts = ptscrop;
                
                filename = sprintf('frame/frame%d.png', frameCount);
                imwrite(im, filename);
                
            else
                box = [0 0];
            end
            [ pts , img ] = ccr_track( im , pts , model );
        end
    else
        % - track the points
         [ imcrop , ptscrop , box ] = cropImage( im , pts );
            if size( imcrop , 1 ) * size( imcrop , 2 ) > 0
                im = imcrop; pts = ptscrop; 
            else
                box = [0 0];
            end
        [ pts , img ] = ccr_track( im , pts , model );
         
        % - detect whether the result is OK or not
        % - here we use the same distance both to detect a lost and to estimate the
        % suitability of a frame to update the model's tracker
        data.distance(i) = detect_lost( img , pts , params );
        threshold = ccr_obtain_threshold( params , pts , model );
        if data.distance(i) < threshold
            % - lost frame, try to detect the face again
            if ispc
               pts = detect_pts_SDM( im , model );
            else
                pts = detect_points_PRL_Linux( im , model.cbSDM_model );
            end
            if np == 68 && size( pts , 1 ) == 66
            pts = [pts(1:60,:); zeros(1,2); pts(61:63,:); zeros(1,2); pts(64:66,:)];
            end
            if ~isempty( pts )
                [ pts , img ] = ccr_track( im , pts , model );
            end
            fprintf( 'Tracker Lost\n');
        end
    end
    
    %%%%% FURTHER PROCESSING HERE
%     if ~isempty( pts )
%         [ rec, imgout ] = registerandwarp(im2double(im), double(pts), model.shModel , [256 256] , 0 );
%     end
    
    % - update the data with the statistics obtained through the given
    % frames
    if ~isempty( pts )
        [ data , model ] = ccr_update_data( pts + ones(np,1)*box , model , data , i );
    else
        data.lost(i) = 1;
        if i == 1
            data.tracked_pts(:,:,i) = -1 * ones( np , 2 );
        else
            data.tracked_pts(:,:,i) = data.tracked_pts(:,:,i-1);
        end
    end
    if incLearning
        % - check suitability
        [suitable,data.counter] = policy_svm( data.distance , i , data.counter );
        if suitable 
            [ model , data ] = iccr_update( img , pts , model , data , i, params.lambda );
        end
    end
    te = toc;
    
    if isempty(pts);
        output.pred = [];
    else
        output.pred = 1 * data.tracked_pts(:,:,i);
    end;
    if params.show && first
        first = false;
        frame_w = size(imcpy,2);
        draw_figure( size(imcpy,2) , size( imcpy, 1 ) );
        set(S.im_h,'cdata',imcpy);
       % drawow;
    end

    
    if params.show
        set(S.im_h,'cdata',imcpy);
        if isempty(output.pred) % if lost/no face, delete all drawings
            if drawed, delete_handlers(); end
        else % else update all drawings
            update_GUI();
        end
    end
    
    drawnow;
    if ~isempty( pts )
        pts = pts + ones(np,1)*box;
        data.tracked_pts(:,:,i) = pts;
    end
    i = i + 1;
    if ~isempty(video)
        if i > obj.NumberOfFrames
            keep = 0;
        end;
    end
    if params.show
        if ~ishandle( S.fh )
            data.tracked_pts = data.tracked_pts(:,:,1:i-1);
            data.lost = data.lost(1:i-1);
            data.distance = data.distance(1:i-1);
            return;
        end;
    end
    frameCount = frameCount+1;
end;

fprintf('\nframeCount = %d \n', frameCount);

if params.show
    close;
end


    function draw_figure( frame_w , frame_h )
               
        
        % 500 -50
        S.fh = figure('units','pixels',...
            'position',[500 50 frame_w frame_h],...
            'menubar','none',...
            'name','CONTINUOUS REGRESSION',...
            'numbertitle','off',...
            'resize','off',...
            'renderer','painters');
        
        % create axes
        S.ax = axes('units','pixels',...
            'position',[1 1 frame_w frame_h],...
            'drawmode','fast');
        
        drawed = false; % nor draw anything yet
        output.pred = [];    % prediction set to null enabling detection
        S.im_h = imshow(zeros(1*frame_h,1*frame_w,3,'uint8'));
        hold on;
        if incLearning
            H = uicontrol('Style', 'PushButton', ...
                'String', 'Deactivate IL', ...
                'Callback',  @(src,evnt)change_String, ...
                'Position',[ 10 10 100 20 ]);
        else
            H = uicontrol('Style', 'PushButton', ...
                'String', 'Activate IL', ...
                'Callback', @(src,evnt)change_String, ...
                'Position',[ 10 10 100 20 ]);
        end;
    end
    function delete_handlers() 
      delete(S.pts_h); delete(S.time_h);
      %delete(S.hh{1}); delete(S.hh{2}); delete(S.hh{3});
      drawed = false;
    end
    function update_GUI()
    
      if drawed % faster to update than to creat new drawings
        
        set(S.pts_h, 'xdata', output.pred(:,1), 'ydata',output.pred(:,2));
        % update frame/second
        set(S.time_h, 'string', sprintf('%d FPS',uint8(1/te)));
      else       
        % create tracked points drawing
 %       S.pts_h   = plot(output.pred(:,1), output.pred(:,2), 'g*', 'markersize',2);
       S.pts_h   = plot(output.pred(:,1), output.pred(:,2), 'r.', 'markersize',16);

        
        % create frame/second drawing
        S.time_h  = text(frame_w-100,50,sprintf('%d FPS',uint8(1/te)),'fontsize',20,'color','m');
        drawed = true;
      end
    end
    function change_String()
        StrVal = get(H,'String');
        if strcmp( StrVal, 'Activate IL')
            set(H,'String','Deactivate IL');
        else
            set(H,'String','Activate IL');
        end;
        
        incLearning = not(incLearning);
    end

   
end

