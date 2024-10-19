%% Imposing Fairness - Fair Recommendation Engine
    
%% Data Pre-Processing (same as before)

Ratings = readcell('MovieLens Dataset.xlsx', 'Sheet','Ratings', 'Range','A2:C100001'); % access the sheet Ratings in the dataset.
Ratings(1:5,:,:); % display the first 5 rows 

numRows = size(Ratings, 1); % size of the matrix (nb of rows)
permutedIndices = randperm(numRows); % permutation of indices

shuffled_Ratings = Ratings(permutedIndices, :); % shuffle the rows of the matrix

isDataUnchanged = isequal(sortrows(Ratings), sortrows(shuffled_Ratings)); % making sure the data didnt change 
if isDataUnchanged
    disp('Data in each row remains the same after shuffling.'); % make sure the rating of user 'i' to movie 'j' is still the same
else
    disp('Data in each row has changed after shuffling.');
end

shuffled_Ratings(1:5,:,:); % the data didnt change and the rows are shuffled
Ratings(1:5,:,:);
Ratings = shuffled_Ratings; % Now the Ratings rows are shuffled and we can split our data into train/test.

% Split into Training and Testing
percentageTraining = 0.8;
numTraining = round(percentageTraining * numRows); %nb of rows of training data

Ratings_training = Ratings(1:numTraining, :); % training matrix/ dataset
Ratings_testing = Ratings(numTraining+1:end, :); % testing matrix/ dataset

Movies = readcell('MovieLens Dataset.xlsx','Sheet','Movies', 'Range','A2:U1683');
Users = readcell('MovieLens Dataset.xlsx', 'Sheet', 'Users', 'Range', 'A2:D944'); % read the Users Sheet

%% Find the IDs for each gender and for each genre of movies
U = rand(943,r); V = rand(1682,r); p = rand(943,1); q = rand(1682,1); r =9;
Users = readcell('MovieLens Dataset.xlsx', 'Sheet', 'Users', 'Range', 'A2:D944'); % read the Users Sheet
malesID = []; femalesID =[]; romanceID =[]; scifiID = []; actionID = []; musicalID =[];

for i = 1:size(Users,1)
    if strcmp(Users(i, 3), 'M') % if it's a male
        malesID = [malesID; cell2mat(Users(i,1))];
    end
    if strcmp(Users(i, 3), 'F') % if it's a female
        femalesID = [femalesID; cell2mat(Users(i,1))];
    end
end

for i = 1:size(Movies,1)
    if cell2mat(Movies(i,17)) == 1 %romance 
        romanceID = [romanceID; cell2mat(Movies(i,1))];
    end
    if cell2mat(Movies(i, 4)) == 1 % action
        actionID = [actionID; cell2mat(Movies(i,1))];
    end
    if cell2mat(Movies(i,18)) ==1 % sci-fi
        scifiID = [scifiID; cell2mat(Movies(i,1))];
    end
    if cell2mat(Movies(i, 15)) %musical
        musicalID = [musicalID; cell2mat(Movies(i,1))];
    end
end
size(malesID)

%% Optimization with new penalty term 
alpha = 0.01;
lambda1 = 1;
lambda2 = 0.01; % for the second regularization term
RMSE_training = zeros(1,100);
counter = 0;
counter2 = 0; %to plot diff in fairness each 100k iterations
iteration = 10000;
iteration2 = 100000;
FairnessList = zeros(1,10);
RMSE_training_index =1;
RMSE_test = zeros(1,100);
RMSE_test_index = 1;
for i=1:1000000
    male_rating = 0; female_rating = 0; cum_male_rating =0; cum_female_rating = 0;
    randomIndex = randi(size(Ratings_training, 1)); % pick random index from the Ratings dataset for SGD
    userID = cell2mat(Ratings_training(randomIndex,1)); % get the index we're going to use for matrix U
    movieID = cell2mat(Ratings_training(randomIndex,2)); % get the index we're going to use for matrix V
    rating = cell2mat(Ratings_training(randomIndex,3)); % get the actual rating 

    error = U(userID,:)*V(movieID,:)' + p(userID) + q(movieID) - rating;
    % Update rules for each
    if ismember(movieID, romanceID)
        if ismember(userID, malesID(1)) % if it's a male, randomly select a female and impose fairness for this specific movie
            male_rating = U(userID, :) * V(movieID, :)' + p(userID) + q(movieID);
            femID = length((femalesID));
            female_rating = U(femalesID(femID), :) * V(movieID, :)' + p(femalesID(femID)) + q(movieID);
        elseif ismember(userID, femalesID)
            female_rating = U(userID, :) * V(movieID, :)' + p(userID) + q(movieID);
            maID = length((malesID)); %random ID for male
            male_rating = U(malesID(maID), :) * V(movieID, :)' + p(malesID(maID)) + q(movieID);
        end
        fairness_term_romance = 2 * (male_rating - female_rating);
        U_update = 2 * (error) * V(movieID, :) + 2 * lambda1 * U(userID, :); 
        if ismember(userID, malesID(1)) %we do this to know what male ID we use in the update rule of V
            V_update = 2 * (error) * U(userID, :) + 2 * lambda1 * V(movieID, :) + 2 * lambda2 * fairness_term_romance * (U(userID, :) - U(femalesID(femID),:));
        else
            V_update = 2 * (error) * U(userID, :) + 2 * lambda1 * V(movieID, :) + 2 * lambda2 * fairness_term_romance * (U(malesID(maID), :) - U(userID,:));
        end
        p_update = 2 * (error) ;
        q_update = 2 * (error) ;
    end


    if ismember(movieID, musicalID)% we're doing this to compute the fairness metric when we obtain a movie of one of the specific genres 
        if ismember(userID, malesID(1))
            male_rating = U(userID, :) * V(movieID, :)' + p(userID) + q(movieID);
            femID = length((femalesID));
            female_rating = U(femalesID(femID), :) * V(movieID, :)' + p(femalesID(femID)) + q(movieID);
        elseif ismember(userID, femalesID(1))
            female_rating = U(userID, :) * V(movieID, :)' + p(userID) + q(movieID);
            maID = length((malesID));
            male_rating = U(malesID(maID), :) * V(movieID, :)' + p(malesID(maID)) + q(movieID);
        end
        fairness_term_musical = 2 * (male_rating - female_rating);
        U_update = 2 * (error) * V(movieID, :) + 2 * lambda1 * U(userID, :);
        if ismember(userID, malesID(1))
            V_update = 2 * (error) * U(userID, :) + 2 * lambda1 * V(movieID, :) + 2 * lambda2 * fairness_term_musical * (U(userID, :) - U(femalesID(femID),:));
        else
            V_update = 2 * (error) * U(userID, :) + 2 * lambda1 * V(movieID, :) + 2 * lambda2 * fairness_term_musical * (U(malesID(maID), :) - U(userID,:));
        end
        p_update = 2 * (error) ;
        q_update = 2 * (error) ;
    end


    if ismember(movieID, scifiID)% we're doing this to compute the fairness metric when we obtain a movie of one of the specific genres 
        if ismember(userID, malesID(1))
            male_rating = U(userID, :) * V(movieID, :)' + p(userID) + q(movieID);
            femID = length((femalesID));
            female_rating = U(femalesID(femID), :) * V(movieID, :)' + p(femalesID(femID)) + q(movieID);
        elseif ismember(userID, femalesID(1))
            female_rating = U(userID, :) * V(movieID, :)' + p(userID) + q(movieID);
            maID = length((malesID));
            male_rating = U(malesID(maID), :) * V(movieID, :)' + p(malesID(maID)) + q(movieID);
        end
        fairness_term_scifi = 2 * (male_rating - female_rating);
        U_update = 2 * (error) * V(movieID, :) + 2 * lambda1 * U(userID, :);
        if ismember(userID, malesID(1))
            V_update = 2 * (error) * U(userID, :) + 2 * lambda1 * V(movieID, :) + 2 * lambda2 * fairness_term_scifi * (U(userID, :)-U(femalesID(femID),:));
        else
            V_update = 2 * (error) * U(userID, :) + 2 * lambda1 * V(movieID, :) + 2 * lambda2 * fairness_term_scifi * (U(malesID(maID), :) - U(userID,:));
        end
        p_update = 2 * (error) ;
        q_update = 2 * (error) ;
    end

    if ismember(movieID, actionID)% we're doing this to compute the fairness metric when we obtain a movie of one of the specific genres 
        if ismember(userID, malesID(1))
            male_rating = U(userID, :) * V(movieID, :)' + p(userID) + q(movieID);
            femID = length((femalesID));
            female_rating = U(femalesID(femID), :) * V(movieID, :)' + p(femalesID(femID)) + q(movieID);
        elseif ismember(userID, femalesID(1))
            female_rating = U(userID, :) * V(movieID, :)' + p(userID) + q(movieID);
            maID = length((malesID));
            male_rating = U(malesID(maID), :) * V(movieID, :)' + p(malesID(maID)) + q(movieID);
        end
        fairness_term_action = 2 * (male_rating - female_rating);
        U_update = 2 * (error) * V(movieID, :) + 2 * lambda1 * U(userID, :);
        if ismember(userID, malesID)
            V_update = 2 * (error) * U(userID, :) + 2 * lambda1 * V(movieID, :) + 2 * lambda2 * fairness_term_action * (U(userID, :)-U(femalesID(femID),:));
        else
            V_update = 2 * (error) * U(userID, :) + 2 * lambda1 * V(movieID, :) + 2 * lambda2 * fairness_term_action * (U(malesID(maID), :) - U(femalesID(femID),:));
        end
        p_update = 2 * (error) ;
        q_update = 2 * (error) ;
    else
        U_update = 2*(error)*V(movieID,:) + 2*lambda1*U(userID,:);
        V_update = 2*(error)*U(userID,:) + 2*lambda1*V(movieID,:);
        p_update = 2*(error);
        q_update = 2*(error);
    end
    % Updating each term 
    U(userID,:) = U(userID,:) - alpha*U_update;
    V(movieID,:) = V(movieID,:) - alpha*V_update;
    p(userID) = p(userID) - alpha*p_update;
    q(movieID) = q(movieID) - alpha*q_update;
    counter = counter + 1;
    counter2 = counter2 +1;
    if counter == iteration
        iteration = iteration + 10000;
        RMSE_training(RMSE_training_index) = getError(U, V, p, q, Ratings_training); % get the training error
        RMSE_training_index = RMSE_training_index + 1; % update the index 
        RMSE_test(RMSE_test_index) = getError(U, V, p, q, Ratings_testing); % testing error
        RMSE_test_index = RMSE_test_index +1;
    end  
    if counter2 == iteration2
        iteration2 = iteration2 + 100000;
        for i = 1:size(Movies, 1)
            G_musical = []; q_musical = [];
            if cell2mat(Movies(i, 15)) == 1 % check if the film is musical
                G_musical = [G_musical; V(i,:)]; %get latent features of musical movies
                q_musical = [q_musical; q(i)];
            end
            % latent features of males and females at this nb of iterations
            U_males =[]; % initialize the vectors
            U_females =[];
            p_males = [];
            p_females = [];
            for i = 1:size(Users, 1) % iterate over the Users matrix
                if strcmp(Users{i, 3}, 'M') % if it's a male, add the the latent features of this male to the vector of males
                    U_males = [U_males; U(i, :)]; % each row will now have the latent features of one male user
                    p_males = [p_males; p(i)];
                end
                if strcmp(Users{i, 3}, 'F') % same procedure for females
                    U_females = [U_females; U(i,:)];
                    p_females = [p_females; p(i)];
                end
            end
            %avg prediction males
            g = G_musical; m2 = size(g,1); q2 = q_musical; % change these if we want to see the avg prediction for another genre 

            m1 = size(U_males,1); cum_rating_males = 0;
            for i=1:m1
                for j = 1:m2
                    R_males = U_males(i,:)*g(j,:)' + p_males(i) + q2(j);
                    cum_rating_males = cum_rating_males + R_males;
                end    
            end
            average_rating_male = cum_rating_males/(m1*m2);
            %avg rating females
            g = G_musical; m2 = size(g,1); q2 = q_musical;
            n1 = size(U_females,1); cum_rating_females = 0;
            for i=1:n1
                for j = 1:m2
                    R_females = U_females(i,:)*g(j,:)' + p_females(i) + q2(j);
                    cum_rating_females = cum_rating_females + R_females;
                end    
            end
            average_rating_female = cum_rating_females/(n1*m2);

            f = (average_rating_male - average_rating_female)^2;
            FairnessList(i) = f;
        end
    end

end

% Plotting RMSE change over iterations
figure;

% Plot RMSE (training)
plot(1:100, RMSE_training, 'LineWidth', 2, 'DisplayName', 'Training');
hold on;

% Plot RMSE (testing)
plot(1:100, RMSE_test, 'LineWidth', 2, 'DisplayName', 'Testing');

title(['RMSE Change for r=' num2str(r) ', lambda=' num2str(lambda)]);
xlabel('Iteration (x10^4)'); % Adjusted xlabel
ylabel('RMSE');
grid on;
legend('Location', 'Best');
ylim([0, 2]);

hold off;

plot(1:10, FairnessList, 'LineWidth',2, 'DisplayName','Change of fairness value w.r.t epochs')

function rmse = getError(U, V, p, q, Data) % we commented this code bcz
                          %it was doing an error for the code after 
    n = size(Data,1);         % if u want to run it, comment everything
                              % below and run
    cum_error = 0;
    for i = 1:n
        user_ID = cell2mat(Data(i,1));
        movie_ID = cell2mat(Data(i,2));
        rating2 = cell2mat(Data(i,3));
        error2 = U(user_ID,:)*V(movie_ID,:)' + p(user_ID) + q(movie_ID) - rating2;
        cum_error = error2^2 + cum_error;
    end
    rmse = sqrt((1/n)*cum_error);
end

%% Get the latent features of males, females, genres

% Users = readcell('MovieLens Dataset.xlsx', 'Sheet', 'Users', 'Range', 'A2:D944'); % read the Users Sheet
% U_males =[]; % initialize the vectors
% U_females =[];
% p_males = [];
% p_females = [];
% for i = 1:size(Users, 1) % iterate over the Users matrix
%     if strcmp(Users{i, 3}, 'M') % if it's a male, add the the latent features of this male to the vector of males
%         U_males = [U_males; U(i, :)]; % each row will now have the latent features of one male user
%         p_males = [p_males; p(i)];
%     end
%     if strcmp(Users{i, 3}, 'F') % same procedure for females
%         U_females = [U_females; U(i,:)];
%         p_females = [p_females; p(i)];
%     end
% end
% U_males(1:5,:); % display some results to see if it's correct
% U_females(1:5,:); % it's correct because we have 9 columns (of the latent features) and Umales & Ufemales together have 943 users
% % Display the results
% disp('Male Users:');
% disp(U_males);
% disp(p_males);
% disp(p_females)
% 
% disp('Female Users:');
% disp(U_females);
% 
% 
% % Make the vectors containing the latent features of  the movies of each
% % genre (same procedure like users but we look here at Movies matrix)
% G_romance = [] ; G_action = []; G_scifi = []; G_musical = []; 
% q_romance = []; q_action = []; q_scifi = []; q_musical = [];
% % Romance is col 17, Action is col 4, Scifi is col 18 and Musical is col 15
% %
% for i = 1:size(Movies, 1)
%     if cell2mat(Movies(i, 17)) == 1 %check if the film is romance
%         G_romance = [G_romance; V(i, :)];
%         q_romance = [q_romance; q(i)];
%     end
%     if cell2mat(Movies(i, 4)) == 1 % check if the film is action
%         G_action = [G_action; V(i, :)]; % add the latent features to the corresponding vector
%         q_action = [q_action; q(i)];
%     end
%     if cell2mat(Movies(i, 18)) == 1 % check if the film is sci-fi
%         G_scifi = [G_scifi; V(i,:)];
%         q_scifi = [q_scifi; q(i)];
%     end
%     if cell2mat(Movies(i, 15)) == 1 % check if the film is musical
%         G_musical = [G_musical; V(i,:)];
%         q_musical = [q_musical; q(i)];
%     end
% end % Now we have vectors containing males, females and the movies of each different genre with their latent features
% % Since here we dont really need the IDs, we didnt include the ID of each
% % user but we could have made a column containing the IDs 
% %% 2- Measuring average predictions for males
% 
% g = G_scifi; m2 = size(g,1); q2 = q_scifi; % change these if we want to see the avg prediction for another genre 
% % Avergae Rating Preditions for males - 
% m1 = size(U_males,1); cum_rating_males = 0;
% for i=1:m1
%     for j = 1:m2
%         R_males = U_males(i,:)*g(j,:)' + p_males(i) + q2(j);
%         cum_rating_males = cum_rating_males + R_males;
%     end    
% end
% average_rating_male = cum_rating_males/(m1*m2);
% disp('Average Predicted Rating for Males');
% disp(average_rating_male);
% 
% %% 3- Measuring average predictions for females
% g = G_scifi; m2 = size(g,1); q2 = q_scifi;
% n1 = size(U_females,1); cum_rating_females = 0;
% for i=1:n1
%     for j = 1:m2
%         R_females = U_females(i,:)*g(j,:)' + p_females(i) + q2(j);
%         cum_rating_females = cum_rating_females + R_females;
%     end    
% end
% average_rating_female = cum_rating_females/(n1*m2);
% disp('Average Predicted Rating for Females');
% disp(average_rating_female);