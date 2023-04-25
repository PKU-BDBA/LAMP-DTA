from TableFileUtils import *


def reg_plot(dz_raw, ax, df, y_pred_col, y_label_col, title, hue=None, xy=True, fit=False, xylim=None, postfix='', root_path=None, size=(3.5, 3.5), color_dict=None, assigned_mae=None, **kwards):
    y_label = df[y_label_col].values
    y_pred = df[y_pred_col].values
    min_value = y_label.min()
    max_value = y_label.max()
    tot, na, valid = len(dz_raw), len(dz_raw[dz_raw[y_label_col].isna()]), len(dz_raw[dz_raw[y_label_col].notna()])  # luozc
    if hue is not None:
        for c, subdf in df.groupby(hue):
            y_label = subdf[y_label_col].values
            y_pred = subdf[y_pred_col].values
            mae = mean_absolute_error(y_label, y_pred)
            mse = mean_squared_error(y_label, y_pred)
            r2 = r2_score(y_label, y_pred)
            pcc = pearsonr(y_label, y_pred)
            sroc = spearmanr(y_label, y_pred)
            # luozc
            if mae > 100:
                label = f'{c}: MAE: {mae:.2E}, R2: {r2:.2f}\nPCC: {pcc[0]:.2f}, Total: {tot}\nNA: {na}, Valid: {valid}'
            else:
                label = f'{c}: MAE={mae:.2f}, R2={r2:.2f}\nPCC: {pcc[0]:.2f}, Total: {tot}\nNA: {na}, Valid: {valid}'
            # luozc
            ax.scatter(y_label, y_pred, label=label, color=color_dict[c], **kwards)
            if fit:  # luozc: 默认不绘制回归线
                m, b = np.polyfit(y_label, y_pred, 1)
                ax.plot(y_label, m*y_label + b, color=color_dict[c])
    else:
        mae = mean_absolute_error(y_label, y_pred)
        mse = mean_squared_error(y_label, y_pred)
        r2 = r2_score(y_label, y_pred)
        pcc = pearsonr(y_label, y_pred)
        sroc = spearmanr(y_label, y_pred)
        if assigned_mae is not None:
            mae = assigned_mae
        # luozc
        if mae > 100:
            label = f'MAE: {mae:.2E}, R2: {r2:.2f}\nPCC: {pcc[0]:.2f}, Total: {tot}\nNA: {na}, Valid: {valid}'
        else:
            label = f'MAE={mae:.2f}, R2={r2:.2f}\nPCC: {pcc[0]:.2f}, Total: {tot}\nNA: {na}, Valid: {valid}'
        # luozc
        ax.scatter(y_label, y_pred, label=label, **kwards)
        if fit:  # luozc: 默认不绘制回归线
            m, b = np.polyfit(y_label, y_pred, 1)
            ax.plot(y_label, m*y_label + b)
    diff = (max_value - min_value) / 16
    if xylim is None:
        xylim = max_value
    # luozc
    if xy:
        ax.plot([min_value - diff, xylim + diff], [min_value - diff, xylim + diff], color='gray', lw=1, linestyle='--')
    ax.legend(loc="upper left")
    # luozc
    ax.set(xlim=(min_value - diff, xylim + diff), ylim=(min_value - diff, xylim + diff))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Actual' + postfix, fontsize=10)
    ax.set_ylabel('AI assessed' + postfix, fontsize=10)
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
#     ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xticks(ax.get_yticks()[1:-1])
#     ax.legend(loc=4, fontsize=11)
    
    l = ax.legend(loc="upper left", frameon=False)
    for legendHandle in l.legendHandles:
        legendHandle.set_alpha(1)
        legendHandle.set_sizes([10])
        
    if root_path is not None:
        ensure_file(f'{root_path}')
        plt.savefig(f'{root_path}', bbox_inches='tight', format='png', dpi=300, facecolor='white', transparent=False)
        remove_alpha_channel(f'{root_path}')
#     plt.show()
#     plt.close(fig)

def reg_plot_forgrid(dz_raw, ax, index, label_reg):
    label = label_reg[index][6:-4]
    df_tmp = dz_raw[dz_raw[f'label_{label}_reg'].notna()].copy()
    # luozc
    # for i in "_sqrt _exp _square _add1log".split():
    #     if i in label:
    #         orig_label = label.replace(i, '')
    #         df_tmp[f'label_{label}_reg'] = df_tmp[f'label_{orig_label}_reg']
    #         if i == '_sqrt':
    #             df_tmp[f'label_{label}_reg_prob_0'] = df_tmp[f'label_{label}_reg_prob_0'].mask(df_tmp[f'label_{label}_reg_prob_0'] < 0, 0)
    #             df_tmp[f'label_{label}_reg_prob_0'] = np.square(df_tmp[f'label_{label}_reg_prob_0'])
    #         elif i == '_exp':
    #             df_tmp[f'label_{label}_reg_prob_0'] = df_tmp[f'label_{label}_reg_prob_0'].mask(df_tmp[f'label_{label}_reg_prob_0'] < 1, 1)
    #             df_tmp[f'label_{label}_reg_prob_0'] = np.log(df_tmp[f'label_{label}_reg_prob_0'])
    #         elif i == '_square':
    #             df_tmp[f'label_{label}_reg_prob_0'] = df_tmp[f'label_{label}_reg_prob_0'].mask(df_tmp[f'label_{label}_reg_prob_0'] < 0, 0)
    #             df_tmp[f'label_{label}_reg_prob_0'] = np.sqrt(df_tmp[f'label_{label}_reg_prob_0'])
    #         else:
    #             df_tmp[f'label_{label}_reg_prob_0'] = df_tmp[f'label_{label}_reg_prob_0'].mask(df_tmp[f'label_{label}_reg_prob_0'] < 0, 0)
    #             df_tmp[f'label_{label}_reg_prob_0'] = (np.exp(df_tmp[f'label_{label}_reg_prob_0'])).apply(lambda x: x - 1)
    #         break
    # luozc
    reg_plot(dz_raw, ax, df_tmp, f'label_{label}_reg_prob_0', f'label_{label}_reg', f'{label}', alpha=0.5, s=3)

def reg_plot_for_one_label(df, one_label):
    dz_raw = df.copy()
    label_reg = [f'label_{one_label}_reg']
    for label in label_reg:
        ori_label = label[6: -4]
        dz_raw[label] = dz_raw[ori_label]
    dz_raw[f'label_{one_label}_reg_prob_0'] = dz_raw[f'{one_label}_pred']

    n_fig = len(label_reg)
    n_col = 5
    n_row = -(-n_fig // n_col)
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4))
    for index in range(n_fig):
        ax = axs.ravel()[index]
        title = reg_plot_forgrid(dz_raw, ax, index, label_reg)
    plt.tight_layout()
    plt.show()
    plt.close(fig)