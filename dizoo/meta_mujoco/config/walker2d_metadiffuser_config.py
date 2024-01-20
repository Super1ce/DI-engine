from easydict import EasyDict

main_config = dict(
    exp_name="walker_params_md_seed0",
    env=dict(
        env_id='walker_params',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        returns_scale=1.0,
        termination_penalty=-100,
        max_path_length=1000,
        use_padding=True,
        include_returns=True,
        normed=False,
        stop_value=8000,
        horizon=32,
        obs_dim=17,
        action_dim=6,
        test_num=10,
    ),
    policy=dict(
        cuda=True,
        max_len=32,
        max_ep_len=200,
        task_num=40,
        train_num=1,
        obs_dim=17,
        act_dim=6,
        no_state_normalize=False,
        no_action_normalize=False,
        need_init_dataprocess=True,
        model=dict(
            diffuser_model_cfg=dict(
                model='DiffusionUNet1d',
                model_cfg=dict(
                    transition_dim=23,
                    dim=64,
                    dim_mults=[1, 4, 8],
                    returns_condition=True,
                    kernel_size=5,
                    attention=False,
                ),
                horizon=32,
                obs_dim=17,
                action_dim=6,
                n_timesteps=20,
                predict_epsilon=False,
                condition_guidance_w=1.2,
                loss_discount=1,
            ),
            reward_cfg=dict(
                model='TemporalValue',
                model_cfg=dict(
                    horizon = 32,
                    transition_dim=23,
                    dim=64,
                    out_dim=32,
                    dim_mults=[1, 4, 8],
                    kernel_size=5,
                    returns_condition=True,
                    no_need_ret_sin=True,
                ),
                horizon=32,
                obs_dim=17,
                action_dim=6,
                n_timesteps=20,
                predict_epsilon=True,
                loss_discount=1,
            ),
            horizon=32,
            n_guide_steps=2,
            scale=0.1,
            t_stopgrad=2,
            scale_grad_by_std=True,
        ),
        normalizer='GaussianNormalizer',
        learn=dict(
            data_path=None,
            train_epoch=60000,
            gradient_accumulate_every=2,
            batch_size=32,
            learning_rate=2e-4,
            discount_factor=0.99,
            learner=dict(hook=dict(save_ckpt_after_iter=1000000000, )),
            eval_batch_size=8,
            warm_batch_size=640,
            test_num=10,
        ),
        collect=dict(data_type='meta_traj', ),
        eval=dict(
            evaluator=dict(
                eval_freq=500, 
                test_env_num=10,
            ),
            test_ret=0.9,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
    ),
    dataset=dict(
        data_dir_prefix='/mnt/nfs/share/meta/walker_traj/buffers_walker_param_train',
        rtg_scale=1,
        context_len=1,
        stochastic_prompt=False,
        need_prompt=False,
        test_id=[5,10,22,31,18,1,12,9,25,38],
        cond=True,
        env_param_path='/mnt/nfs/share/meta/walker/env_walker_param_train_task',
        need_next_obs=True,
    ),
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='meta',
        import_names=['dizoo.meta_mujoco.envs.meta_env'],
    ),
    env_manager=dict(type='meta_subprocess'),
    policy=dict(
        type='metadiffuser',
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)
create_config = create_config