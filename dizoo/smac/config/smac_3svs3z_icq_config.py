from easydict import EasyDict

agent_num = 3
evaluator_env_num = 8

main_config = dict(
    exp_name='smac_3svs_3zcoma_seed0',
    env=dict(
        map_name='3s_vs_3z',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        obs_last_action=False,
        evaluator_env_num=evaluator_env_num,
        stop_value=0.99,
        n_evaluator_episode=16,
        death_mask=True,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        model=dict(
            agent_num=agent_num,
            obs_shape=36,
            action_shape=9,  # action_one_hot size
            actor_hidden_size_list=[64],
            global_obs_shape=27,
            embedding_size=32,
            critic_hidden_size=256,
        ),
        learn=dict(
            train_epoch=1000,
            learning_rate_critic=5e-4,
            learning_rate_policy=5e-4,
            discount_factor=0.99,
            clip_value=20,
            td_lambda=0.75,
            target_update_theta=0.008,
            batch_size=32,
        ),
        # used in state_num of hidden_state
        collect=dict(data_type='icq', data_path='/mnt/nfs/zhaochen/ICQ/ICQ-MA/3s_vs_3z.h5'),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=10, )),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='icq'),
)
create_config = EasyDict(create_config)

if __name__ == '__main__':

    from ding.entry import serial_pipeline_offline
    serial_pipeline_offline((main_config, create_config), seed=0)
