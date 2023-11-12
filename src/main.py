"""
The main file to run BSDE solver to solve McKean-Vlasov Forward-Backward Stochastic Differential Equations.

"""

import json
import munch
import os
import logging

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf

import equation as eqn
from solver import SineBMSolver, SineBMDBDPSolver, FlockSolver


flags.DEFINE_string('config_path', 'configs/sinebm_d2.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
                    """The name of numerical experiments, prefix for logging""")
flags.DEFINE_string('logging_dir', '../logs',
                    """Directory where to write event logs and output array""")
FLAGS = flags.FLAGS


def main(argv):
    del argv
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)

    if not os.path.exists(FLAGS.logging_dir):
        os.mkdir(FLAGS.logging_dir)
    path_prefix = os.path.join(FLAGS.logging_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    logging.info('Experiment name: %s' % FLAGS.exp_name)
    if config.eqn_config.eqn_name == "SineBM":
        if config.net_config.loss_type == "DBDPiter":
            bsde_solver = SineBMDBDPSolver(config, bsde)
        else:
            bsde_solver = SineBMSolver(config, bsde)
    elif config.eqn_config.eqn_name == "Flocking":
        bsde_solver = FlockSolver(config, bsde)
    result = bsde_solver.train()
    if config.eqn_config.eqn_name == "Flocking":
        result_str = "R2: {}, y2_err: {}\n".format(result["R2"], result["y2_err"]) + \
            "v_std:\n" + np.array2string(result["v_std"], max_line_width=10000, separator=',', formatter={'float_kind':lambda x: "%.6f" % x}) + "\n"
        np.savetxt('{}_result.txt'.format(path_prefix),
            result["history"],
            fmt=['%d', '%.5e', '%.5e', '%d'],
            delimiter=",",
            header=result_str+'step,loss_function,err_Y2_init,elapsed_time',
            comments='')
        np.savez('{}_path.npz'.format(path_prefix), state_path=result["state_path"])
    elif config.eqn_config.eqn_name == "SineBM":
        result_str = "err_mean_y_final_valid: {}\n".format(result["err_mean_y"]) + \
            "estimated_mean_y:\n" + np.array2string(result["estimated_mean_y"], max_line_width=10000, separator=',', formatter={'float_kind':lambda x: "%.6f" % x}) + "\n" + \
            "true_mean_y:\n" + np.array2string(result["true_mean_y"], max_line_width=10000, separator=',', formatter={'float_kind':lambda x: "%.6f" % x}) + "\n"
        np.savetxt('{}_result.txt'.format(path_prefix),
            result["history"],
            fmt=['%d', '%.5e', '%.5e', '%.5e', '%d'],
            delimiter=",",
            header=result_str+'step,loss_function,Y0_init,err_mean_y,elapsed_time',
            comments='')


if __name__ == '__main__':
    app.run(main)
