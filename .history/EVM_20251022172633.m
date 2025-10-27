%% Reproduce EVM-style demo figure (no toolboxes)
clear; close all; clc; rng(7);

% ---------- 1) 合成四类二维数据 ----------
% 类中心（大致对应图中位置）
C_star  = [-1.15  1.55];   % 红色星
C_diam  = [ 1.25  1.05];   % 绿色菱形
C_square= [ 1.05 -0.20];   % 蓝色方块
C_dot   = [-0.20  0.00];   % 蓝色小点（被 A 标注影响的那簇）

% 协方差（控制每簇形状）
S = @(sx,sy,rho)[sx^2 rho*sx*sy; rho*sx*sy sy^2];
Sig_star   = S(0.18,0.20, 0.05);
Sig_diam   = S(0.22,0.18,-0.10);
Sig_square = S(0.20,0.16, 0.05);
Sig_dot    = S(0.25,0.18, 0.00);

% 从高斯采样（不用 mvnrnd）
sample = @(N,mu,SIG) (randn(N,2)*chol(SIG,'lower') + mu);

X_star   = sample(12,C_star,   Sig_star);
X_diam   = sample(30,C_diam,   Sig_diam);
X_square = sample(28,C_square, Sig_square);
X_dot    = sample(45,C_dot,    Sig_dot);

% 加一个“离群星”靠近 A（见原图左中）
X_star = [X_star; 0.02 0.15];

% ---------- 2) 底面散点与样式 ----------
figure('Color','w','Position',[100,100,800,600]);
hold on; box on; grid on;
z0 = 0; % 所有散点都放在 z=0 的平面

scatter3(X_star(:,1),  X_star(:,2),  z0+zeros(size(X_star,1),1),  70, ...
        'pentagram', 'filled', 'MarkerEdgeColor','k', 'MarkerFaceColor',[0.635294 0.0784314 0.184314], 'LineWidth',1.0); % 红色五角星 #A2142F
scatter3(X_diam(:,1),  X_diam(:,2),  z0+zeros(size(X_diam,1),1),  60, ...
        'd', 'filled', 'MarkerEdgeColor','k', 'MarkerFaceColor',[0.466667 0.674510 0.188235]); % 绿色菱形 #77AC30
scatter3(X_square(:,1),X_square(:,2),z0+zeros(size(X_square,1),1),60, ...
        's', 'filled', 'MarkerEdgeColor','k', 'MarkerFaceColor',[0 0.447059 0.741176]); % 蓝色正方形 #0072BD
scatter3(X_dot(:,1),   X_dot(:,2),   z0+zeros(size(X_dot,1),1),   18, ...
        'o', 'filled', 'MarkerEdgeColor','k', 'MarkerFaceColor',[0.929412 0.694118 0.125490]); % 黄色圆形 #EDB120

        

% ---------- 3) ψ-model 等值环（彩色同心） ----------
% 采用带形状参的径向函数 psi(r) = exp(-(r/s)^t)
psi = @(X,Y,cx,cy,s,t) exp( - ( sqrt((X-cx).^2 + (Y-cy).^2) ./ s ).^t );

% 选几处“极端向量/代表点”（包含 A 所在）
EV = [ -0.23 -0.02 ;   % A 对应蓝点极端向量
        1.06 -0.22 ;   % 蓝方块
        1.28  1.03 ;   % 绿菱形
       -1.05  1.55 ];  % 红星
scale = [0.42, 0.36, 0.40, 0.48];   % 各自 scale
shape = [2.6,  2.2,  2.4,  2.0];    % 各自 shape

% 网格
[xg,yg] = meshgrid(linspace(-3,3,400));
colormap(jet); caxis([0 1]);

% 等高线层级（红到蓝：高到低）
levels = linspace(1.0,0.05,12);

for k = 1:size(EV,1)
    Z = psi(xg,yg,EV(k,1),EV(k,2),scale(k),shape(k));
    [~, h] = contour(xg,yg,Z,levels);    % 直接画在底面，保持颜色映射
    set(h,'LineWidth',1.25);

    % 将等高线明确压到 z=0（兼容不同 MATLAB 版本）
    ch = get(h,'Children');
    for ic = 1:numel(ch)
        zd = get(ch(ic),'ZData');
        if ~isempty(zd)
            set(ch(ic),'ZData',zeros(size(zd))); % 强制贴在 z=0
        end
    end
end

% 标出 A 与问号
text(-0.34,-0.06,0.02,'A','FontWeight','bold','FontSize',20,'Color','k');
text( 0.25, 0.35,0.03,'??','FontWeight','bold','FontSize',22,'Color','k');
text( 2.90,-0.05,0.03,'?','FontWeight','bold','FontSize',22,'Color','k');

% ---------- 4) 上方半透明“钟形面”（表示类支持/概率） ----------
% 用四个高斯穹顶，中心与各簇一致
gauss2 = @(X,Y,cx,cy,s) exp(-((X-cx).^2 + (Y-cy).^2)/(2*s^2));
tops  = {C_star, C_diam, C_square, C_dot};
sigT  = [0.65 0.55 0.55 0.60];
amp   = [1.05 0.95 0.90 0.85];

% 抬高穹顶，并缩小每个穹顶的覆盖范围（为每个穹顶使用局部网格）
zlift      = 0.18;   % 统一抬高量（不改变颜色映射）
spanFactor = 1.3;    % 每个穹顶半宽 = spanFactor * sigma（越小覆盖越小）
gridN      = 121;    % 局部网格分辨率

for i=1:4
    cx = tops{i}(1); cy = tops{i}(2);
    span = spanFactor * sigT(i);
    [xgi,ygi] = meshgrid(linspace(cx - span, cx + span, gridN), ...
                         linspace(cy - span, cy + span, gridN));

    Zt = amp(i) * gauss2(xgi,ygi,cx,cy,sigT(i));
    % 用 Zt 作为颜色，Zt+zlift 作为高度，从而抬高且保留原色
    s1 = surf(xgi,ygi, Zt + zlift, Zt, 'EdgeColor','none', 'FaceAlpha',0.5);
end
shading interp; camlight headlight; lighting gouraud;

% ---------- 5) 轴、视角与观感 ----------
xlim([-2 2]); ylim([-1 3]); zlim([0 1.25]); view(-67,22);
axis vis3d; daspect([1 1 0.8]);
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1);
% xlabel('x'); ylabel('y'); zlabel('score / \psi');

% 网格和背景风格
grid on; 
% set(gca,'GridAlpha',0.2,'MinorGridAlpha',0.2);
