%% create input and model
means = [-30 30; 0 0];
Ntr = 1000;

dotnoise = 70;

maxRT = 2.5;
dt = 0.093;

targetinds = randi(2, 1, Ntr);
Dots = bsxfun( @plus, permute(means(:, targetinds), [1,3,2]), ...
    dotnoise * randn(2, round(maxRT / dt), Ntr) );

model = StaticGaussModel(1:2, maxRT, Dots, [0 Inf]);
model.dt = dt;
model.means = means;

model.params.noisestd = 50;
model.params.intstd = 70;
model.params.prior = .5;

model.params.discount = 1;

model.params.Binit = 0.8;
model.params.Bshape = 1.4;
model.params.Bstretch = 0;

model.params.NDTmean = 0;
model.params.NDTstd = 0;

model.params.lapseprob = 0.1;
model.params.lapsetoprob = 0.3;
model.params.lapseRTlims = [0 maxRT];


%% compute posteriors and surprise
nrep = 3;
[logPost, surprise, respRT] = model.genEvidenceFromInput(1:Ntr, nrep);


%% plot example
tr = 3;

figure

cols = [102,194,165; 252,141,98; 141,160,203; 231,138,195] / 255;

% posterior probabilities
subplot(3,1,1)
hold on
for r = 1:nrep
    h = plot( model.Time, exp(logPost(:, :, tr, r))', 'Color', ...
        cols(r, :), 'LineWidth', 2 );
    set(h(2), 'LineStyle', '--')
end
h(3) = h(targetinds(tr));
legend(h, [num2cellstr(1:2, 'alternative %d'), {'correct alt.'}], 'Location', 'East')
legend boxoff
ylabel('posterior probabilities')
title(sprintf('trial %d', tr))
xlim([0 maxRT])
set(gca, 'ColorOrder', cols)
plot(ones(2, 1) * permute(respRT(2, tr, :), [1 3 2]), ylim, ':')
ylim([-0.01 1.01])

% log posterior odds
subplot(3,1,2)
set(gca, 'ColorOrder', cols)
hold on
plot( model.Time, permute( logPost(1, :, tr, :) - logPost(2, :, tr, :), ...
    [2 4 1 3] ), 'LineWidth', 2 )
ylabel('log posterior odds')
xlim([0 maxRT])
legend(num2cellstr(1:nrep, 'rep %d'))
legend boxoff
plot(ones(2, 1) * permute(respRT(2, tr, :), [1 3 2]), ylim, ':')

% surprise
subplot(3,1,3)
set(gca, 'ColorOrder', cols)
hold on
plot( model.Time, permute( surprise(:, tr, :), [1 3 2]), 'LineWidth', 2 )
ylabel('surprise')
xlabel('time (s)')
xlim([0 maxRT])
h = plot(ones(2, 1) * permute(respRT(2, tr, :), [1 3 2]), ylim, ':');
legend(h, num2cellstr(1:nrep, 'DT of rep %d'))
legend boxoff


%% does surprise go up or down on average? - depends on params
%  for low noise in input, i.e., low dotnoise+noisestd surprise will go
%  down on average, but will go up for large input noise
figplot(model.Time, squeeze(mean(mean(surprise, 2), 3)), 'Color', ...
    cols(1, :), 'LineWidth', 2)