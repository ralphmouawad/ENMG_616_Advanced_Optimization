%% ENMG 616 - Advanced Optimization Techniques & Algorithms                     
    % Final Project - Fair Movie Recommendation System 
    % By - Ralph Mouawad & Jade Chabbouh
    % To - Dr. Maher Nouiehed 

%%  Data Pre-Processing

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

%% Sorting IDs between males, females, and the 4 different genres 
Movies = readcell('MovieLens Dataset.xlsx', 'Sheet','Movies', 'Range','A2:U1683'); % access the sheet and read it
malesID = []; femalesID = [];
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

%% Optimization & Evaluation using RMSE 

Ratings_training; Ratings_testing; % our training & testing datasets
r = 9; % maximum number of features to be extracted
alpha = 0.01; % step size
lambda = 1; % regularization

% Initialize the matrices 
U = rand(943,r);
V = rand(1682,r);
p = rand(943,1);
q = rand(1682,1);
RMSE_training = zeros(1,100);
counter = 0;
iteration = 10000;
RMSE_training_index =1;
RMSE_test = zeros(1,100);
RMSE_test_index = 1;
% Using the Stochastic Gradient Descent algorithm (choose 1 random index)
for i=1:1000000
    randomIndex = randi(size(Ratings_training, 1)); % pick random index from the Ratings dataset for SGD
    userID = Ratings_training(randomIndex,1); % get the index we're going to use for matrix U
    movieID = Ratings_training(randomIndex,2); % get the index we're going to use for matrix V
    rating = Ratings_training(randomIndex,3); % get the actual rating 
    
    userID = cell2mat(userID); % the data extracted was in a cell format we converted them into numerical
    movieID = cell2mat(movieID);
    rating = cell2mat(rating);

    % Update rules for each
    error = U(userID,:)*V(movieID,:)' + p(userID) + q(movieID) - rating;
    U_update = 2*(error)*V(movieID,:) + 2*lambda*U(userID,:);
    V_update = 2*(error)*U(userID,:) + 2*lambda*V(movieID,:);
    p_update = 2*(error);
    q_update = 2*(error);

    % Updating each term 
    U(userID,:) = U(userID,:) - alpha*U_update;
    V(movieID,:) = V(movieID,:) - alpha*V_update;
    p(userID) = p(userID) - alpha*p_update;
    q(movieID) = q(movieID) - alpha*q_update;

    counter = counter + 1;
    
    if counter == iteration
        iteration = iteration + 10000;
        RMSE_training(RMSE_training_index) = getError(U, V, p, q, Ratings_training); % get the training error
        RMSE_training_index = RMSE_training_index + 1; % update the index 
        RMSE_test(RMSE_test_index) = getError(U, V, p, q, Ratings_testing); % testing error
        RMSE_test_index = RMSE_test_index +1;
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


% function rmse = getError(U, V, p, q, Data) % we commented this code bcz
% 
%     n = size(Data,1);         % it was doing an error for the next ones.
%                       If you want to run it, comment evrthg below. 
% 
%     cum_error = 0;
%     for i = 1:n
%         user_ID = cell2mat(Data(i,1));
%         movie_ID = cell2mat(Data(i,2));
%         rating2 = cell2mat(Data(i,3));
%         error2 = U(user_ID,:)*V(movie_ID,:)' + p(user_ID) + q(movie_ID) - rating2;
%         cum_error = error2^2 + cum_error;
%     end
%     rmse = sqrt((1/n)*cum_error);
% end



%% Feature Extraction

V; % We are using the one with r = 9 and lambda = 1 bcz it had the best performance

% Let's extract the sheet Movies from our dataset.
Movies = readcell('MovieLens Dataset.xlsx', 'Sheet','Movies', 'Range','A2:U1683'); % access the sheet and read it
Movies(1:5,1:3); % display some films to make sure the file is well extracted
Genre = readcell('MovieLens Dataset.xlsx', 'Sheet','Genres','Range','A2:B20');

featureIndex = 4; % Feature index to analyze
numTopMovies = 10; % Number of top movies to extract for each feature
minRatings = 25; % Minimum ratings for accurate analysis

% Sort movies based on Feature Index of the matrix V in descending order
[~, sortedMovieIndices] = sort(V(:, featureIndex), 'descend');

% Select top movies with at least 25 ratings
topMovies = [];
for movieIndex = sortedMovieIndices' % loop over the sorted indices
    movieID = Movies{movieIndex, 1}; % select the movie 
    % Check if the movie has at least 25 ratings
    if sum(cell2mat(Ratings(:, 2)) == movieID) >= minRatings % if the film has at leat 25 ratings
        topMovies = [topMovies; Movies(movieIndex, 2)]; % Add the movie to the list
        if size(topMovies, 1) == numTopMovies
            break; % Stop when we have enough top movies
        end
    end
end

% Display the results
disp(['Top movies for Feature ' num2str(featureIndex) ':']);
disp(topMovies)

% Get the genre IDs of each movie

% Create a cell array to store genres for each top movie
topGenresList = cell(numTopMovies, 1);

% Iterate over the topMovies list
for i = 1:numTopMovies
    movieName = topMovies{i}; % Get the movie name from the topMovies list

    % Find the row index of the movie in the Movies matrix
    movieRowIndex = find(strcmp(Movies(:, 2), movieName));

    % Initialize a list to store genre IDs for the current movie
    currentGenres = [];

    % Check each genre column (column 3 to 21)
    for genreIndex = 3:21
        if cell2mat(Movies(movieRowIndex, genreIndex)) == 1
            % If the entry is 1, it means the genre is associated with the movie
            currentGenres = [currentGenres, genreIndex - 2]; % Store the genre ID
        end
    end

    % Store the genre IDs for the current movie in the topGenresList
    topGenresList{i} = currentGenres;
end

% Display the genre IDs for each top movie
disp('Genre IDs for Each Top Movie:');
disp(topGenresList); % we got the genre ID for each film (check report for their names).

%% Recommending Similar Movies -
selectedFilmIndex = 500; % choose the movie we want to find similar ones to

% Check if the selected movie has at least 25 ratings, if not we have to
% select another one
if sum(cell2mat(Ratings(:, 2)) == selectedFilmIndex) < 25
    disp(['The selected movie (Index ' num2str(selectedFilmIndex) ') does not have at least 25 ratings.']);
    return; % Exit the code if the condition is not met
else
    disp(['The selected movie (Index ' num2str(selectedFilmIndex) ') has 25 or more ratings and is relevant for analysis.']);
end
% We've re-run the code to get a new vector V bcz some columns were sorted
V; % The movie matrix with latent features 


selectedFilmIndex = 500; % choose the movie we want to find similar ones to
v_i = V(selectedFilmIndex, :); % take the features of this movie from the movie matrix V

%Initiate a list of similar movies for 'i'
similarMoviesList = [];


for j = 1:size(V, 1) % Loop over the matrix V
    if j ~= selectedFilmIndex % make sure we dont compare the same movies since they'll be very similar with cos =1
        % Calculate the cosine similarity between vi and vj
        similarity_ij = dot(v_i, V(j, :)) / (norm(v_i) * norm(V(j, :)));
        similarMoviesList = [similarMoviesList; j, similarity_ij]; % add the movie j with tis simil value to the list (now has 2 col)
    end
end

%Sort the list by descending order of similarity values     
similarMoviesList = sortrows(similarMoviesList, -2);

% Make the list called 'topSimilarMovies' by selecting the first 5 movie IDs
topSimilarMovies = similarMoviesList(1:5, 1);

% Display the results
disp(['Top 5 Similar Movies to Selected Movie (Index ' num2str(selectedFilmIndex) '):']);
disp(topSimilarMovies);

% Access the names of the top similar movies
topSimilarMovieNames = Movies(topSimilarMovies, 2);

% Access the name of the selected movie
selectedMovieName = Movies(selectedFilmIndex, 2);

% Display the results
disp(['Selected Movie Name: ' selectedMovieName]);
disp('Top 5 Similar Movies:');
disp(topSimilarMovieNames);

% check the interpretation of the results in the report.

%% Fair Recommendation Engine - 
% In this part we will first measure fairness then impose fairness in a
% different code 


%% 1 - Set matrices containing males & females, & matrices with movie IDs of each genre

% We will define 2 matrices containing males latent features and females
% latent features, and 2 vectors p for error

Users = readcell('MovieLens Dataset.xlsx', 'Sheet', 'Users', 'Range', 'A2:D944'); % read the Users Sheet
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
U_males(1:5,:); % display some results to see if it's correct
U_females(1:5,:); % it's correct because we have 9 columns (of the latent features) and Umales & Ufemales together have 943 users
% Display the results
disp('Male Users:');
disp(U_males);
disp(p_males);
disp(p_females)

disp('Female Users:');
disp(U_females);


% Make the vectors containing the latent features of  the movies of each
% genre (same procedure like users but we look here at Movies matrix)
G_romance = [] ; G_action = []; G_scifi = []; G_musical = []; 
q_romance = []; q_action = []; q_scifi = []; q_musical = [];
% Romance is col 17, Action is col 4, Scifi is col 18 and Musical is col 15
%
for i = 1:size(Movies, 1)
    if cell2mat(Movies(i, 17)) == 1 %check if the film is romance
        G_romance = [G_romance; V(i, :)];
        q_romance = [q_romance; q(i)];
    end
    if cell2mat(Movies(i, 4)) == 1 % check if the film is action
        G_action = [G_action; V(i, :)]; % add the latent features to the corresponding vector
        q_action = [q_action; q(i)];
    end
    if cell2mat(Movies(i, 18)) == 1 % check if the film is sci-fi
        G_scifi = [G_scifi; V(i,:)];
        q_scifi = [q_scifi; q(i)];
    end
    if cell2mat(Movies(i, 15)) == 1 % check if the film is musical
        G_musical = [G_musical; V(i,:)];
        q_musical = [q_musical; q(i)];
    end
end % Now we have vectors containing males, females and the movies of each different genre with their latent features
% Since here we dont really need the IDs, we didnt include the ID of each
% user but we could have made a column containing the IDs 
%% 2- Measuring average predictions for males

g = G_musical; m2 = size(g,1); q2 = q_musical; % change these if we want to see the avg prediction for another genre 
% Avergae Rating Preditions for males - 
m1 = size(U_males,1); cum_rating_males = 0;
for i=1:m1
    for j = 1:m2
        R_males = U_males(i,:)*g(j,:)' + p_males(i) + q2(j);
        cum_rating_males = cum_rating_males + R_males;
    end    
end
average_rating_male = cum_rating_males/(m1*m2);
disp('Average Predicted Rating for Males');
disp(average_rating_male);

%% 3- Measuring average predictions for females
g = G_musical; m2 = size(g,1); q2 = q_musical;
n1 = size(U_females,1); cum_rating_females = 0;
for i=1:n1
    for j = 1:m2
        R_females = U_females(i,:)*g(j,:)' + p_females(i) + q2(j);
        cum_rating_females = cum_rating_females + R_females;
    end    
end
average_rating_female = cum_rating_females/(n1*m2);
disp('Average Predicted Rating for Females');
disp(average_rating_female);

%% Actual ratings of males and females

avgMaleAction = 0;
avgMaleRomance = 0;
avgMaleScifi = 0;
avgMaleMusical = 0;

avgFemaleAction = 0;
avgFemaleRomance = 0;
avgFemaleScifi = 0;
avgFemaleMusical = 0;

totalMaleAction = 0;
totalMaleRomance = 0;
totalMaleScifi = 0;
totalMaleMusical = 0;

totalFemaleAction = 0;
totalFemaleRomance = 0;
totalFemaleScifi = 0;
totalFemaleMusical = 0;

for i = 1:length(Ratings)
    if ismember(Ratings{i, 1}, malesID)
        if ismember(Ratings{i, 2}, romanceID)
            avgMaleRomance = avgMaleRomance + Ratings{i, 3};
            totalMaleRomance = totalMaleRomance + 1;
        elseif ismember(Ratings{i, 2}, actionID)
            avgMaleAction = avgMaleAction + Ratings{i, 3}; 
            totalMaleAction = totalMaleAction + 1;
        elseif ismember(Ratings{i, 2}, scifiID)
            avgMaleScifi = avgMaleScifi + Ratings{i, 3};
            totalMaleScifi = totalMaleScifi + 1;
        elseif ismember(Ratings{i, 2}, musicalID)
            avgMaleMusical = avgMaleMusical + Ratings{i, 3};
            totalMaleMusical = totalMaleMusical + 1;
        end
    elseif ismember(Ratings{i, 1}, femalesID)
        if ismember(Ratings{i, 2}, romanceID)
            avgFemaleRomance = avgFemaleRomance + Ratings{i, 3};
            totalFemaleRomance = totalFemaleRomance + 1;
        elseif ismember(Ratings{i, 2}, actionID)
            avgFemaleAction = avgFemaleAction + Ratings{i, 3}; 
            totalFemaleAction = totalFemaleAction + 1;
        elseif ismember(Ratings{i, 2}, scifiID)
            avgFemaleScifi = avgFemaleScifi + Ratings{i, 3};
            totalFemaleScifi = totalFemaleScifi + 1;
        elseif ismember(Ratings{i, 2}, musicalID)
            avgFemaleMusical = avgFemaleMusical + Ratings{i, 3};
            totalFemaleMusical = totalFemaleMusical + 1;
        end
    end
end

avgFemaleMusical/totalFemaleMusical;


