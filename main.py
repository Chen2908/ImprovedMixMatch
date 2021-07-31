from datetime import datetime
import tqdm
from tensorflow.keras.metrics import Mean, Accuracy
from tensorflow.keras.metrics import Precision, AUC, TrueNegatives, TruePositives, FalsePositives, FalseNegatives, Recall
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from baseline import Baseline
from model import WideResNet
from mixmatch import *
from datasets_load import *
import time
import gc
import argparse
import warnings
from tensorflow.keras.optimizers import Adam
import pandas as pd
from bayes_opt import BayesianOptimization
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
warnings.filterwarnings('ignore')


def main(arguments):
    """
    This function performs the main training loop. It divides the data into train and test 10 times using
    10-fold cross validation and then performs hyperparameters optimizations using Bayesian Optimization
    via 3 fold cross validations.
    :param arguments: main arguments.
    Writes the performance results into files.
    """
    print(datetime.now())
    dataset_name = arguments.dataset
    print(dataset_name)

    # read datasets
    file_path = dataset_name + '.tfrecord'
    x_ds = load_tfrecord_dataset(file_path)  # load x dataset
    y_ds = [int(item['label']) for item in iter(x_ds)]  # labels

    num_classes = 5 if 'cifar100' in dataset_name else 5
    image_size = 32

    models = ['improved_mix_match', 'baseline', 'original_mix_match']

    all_results = {'Dataset name': [], 'Algorithm name': [],
                   'Fold num': [], 'Hyperparameters values': [],
                   'Accuracy': [], 'TPR': [],
                    'FPR': [], 'Precision': [],
                    'AUC': [], 'PR Curve': [],
                    'Training time': [], 'Inference time': []}
    all_AUC_results = {model: [] for model in models}
    # outer loop - 10 fold cross validation
    cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    fold = 1
    x_placeholder = np.zeros(len(y_ds))
    args = initial_const_hyperparameters()
    for train_val_indices, test_indices in cv_outer.split(x_placeholder, y_ds):
        train_val_samples = []
        test_samples = []
        for samples_idx, samples in enumerate(iter(x_ds)):
            if samples_idx in test_indices:
                test_samples.append(samples)
            elif samples_idx in train_val_indices:
                train_val_samples.append(samples)
        test_ds = list_to_tf_dataset_string(test_samples)
        train_val_ds = list_to_tf_dataset_string(train_val_samples)
        test_y = [int(item['label']) for item in iter(test_ds)]  # labels
        train_val_y = [int(item['label']) for item in iter(train_val_ds)]

        for model in models:
            print(model)
            if 'mix_match' in model:
                # call inner loop with train_val_ds
                def wrapper_run_inner_loop(alpha, lambda_u):
                    args['alpha'] = alpha
                    args['lambda_u'] = int(lambda_u)
                    return run_inner_loop(train_val_ds, num_classes, args, train_val_y, model, image_size)

                bayesian_opt = BayesianOptimization(
                    f=wrapper_run_inner_loop, pbounds={'alpha': (0.7, 0.9), 'lambda_u': (75, 115)},
                    random_state=1, verbose=2)

                bayesian_opt.maximize(n_iter=45, init_points=5)
                best_hyper_params = bayesian_opt.max['params']
                args['lambda_u'] = int(best_hyper_params['lambda_u'])
                args['alpha'] = best_hyper_params['alpha']
                train_val_samples = list_to_tf_dataset_string(train_val_samples)
                train_labeled, train_unlabeled = split_labeled_unlabeled_overlap(train_val_ds, 100, test_ds,
                                                                                 num_classes)
                train_x = process_parsed_dataset(train_labeled, num_classes, model)
                train_u = process_parsed_dataset(train_unlabeled, num_classes, model)
                test_ds_process = process_parsed_dataset(test_ds, num_classes, model)

                # train
                print(f'--- train {fold} fold - {model} ---')
                start_train_time = time.time()
                mix_match_model, ema_model = train_mixmatch(num_classes, args, train_x, train_u, model, test_ds_process)
                end_train_time = time.time()
                train_time = end_train_time - start_train_time
                print(f'train time mixmatch:{train_time}')
                start_inference_time = time.time()

                # inference
                print(f'--- predict {fold} fold - {model} ---')
                test_xe_avg, test_results = validate(test_ds_process, ema_model, args, split='test')
                end_inference_time = time.time()
                inference_time = ((end_inference_time - start_inference_time)*1000)/len(test_indices)
                print(f'inference time mixmatch: {inference_time}')

            else:
                # call inner loop with train_val_ds
                def wrapper_run_inner_loop(lr, dropout_rate, dense_size):
                    args['lr'] = lr
                    args['dropout_rate'] = dropout_rate
                    args['dense_size'] = int(dense_size)
                    print('optimization iter')
                    return run_inner_loop(train_val_ds, num_classes, args, train_val_y, model, image_size)

                bayesian_opt = BayesianOptimization(
                    f=wrapper_run_inner_loop, pbounds={'lr': (0.00001, 0.001), 'dropout_rate': (0.1, 0.3),
                                                       'dense_size': (512, 2048)}, random_state=1, verbose=2)
                bayesian_opt.maximize(n_iter=45, init_points=5)
                best_hyper_params = bayesian_opt.max['params']
                args['lr'] = best_hyper_params['lr']
                args['dropout_rate'] = best_hyper_params['dropout_rate']
                args['dense_size'] = int(best_hyper_params['dense_size'])

                train_val_samples_process = process_parsed_dataset(train_val_samples, num_classes, model)
                test_ds_process = process_parsed_dataset(test_ds, num_classes, model)
                train_val_samples_process = train_val_samples_process.shuffle(buffer_size=100).batch(args['batch_size'])
                test_ds_process = test_ds_process.shuffle(buffer_size=100).batch(args['batch_size'])

                print(f'---train {fold} fold---')
                start_train_time = time.time()
                baseline_model, baseline = train_baseline(num_classes, args, train_val_samples_process, image_size)
                end_train_time = time.time()
                train_time = end_train_time - start_train_time
                print(f'train time baseline: {train_time}')
                start_inference_time = time.time()

                # inference
                print(f'---predict {fold} fold---')
                test_results = validate_baseline(baseline, test_ds_process)
                end_inference_time = time.time()
                inference_time = ((end_inference_time - start_inference_time)*1000)/len(test_indices)
                print(f'inference time baseline: {inference_time}')

            all_results['Dataset name'].append(dataset_name)
            all_results['Algorithm name'].append(model)
            all_results['Fold num'].append(fold)
            all_results['Hyperparameters values'].append(best_hyper_params)
            all_results['Accuracy'].append(test_results['Accuracy'])
            all_results['TPR'].append(test_results['TPR'])
            all_results['FPR'].append(test_results['FPR'])
            all_results['Precision'].append(test_results['Precision'])
            all_results['AUC'].append(test_results['AUC'])
            all_results['PR Curve'].append(test_results['AUPRC'])
            all_results['Training time'].append(train_time)
            all_results['Inference time'].append(inference_time)
            all_AUC_results[model].append(test_results['AUC'])

            acc = test_results['Accuracy']
            print(f'accuracy for {model}: {acc}')
        fold += 1
        gc.collect()

    res_df = pd.DataFrame.from_dict(all_results)
    dt = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    if not os.path.exists('results/'):
        os.makedirs('results/')
    res_df.to_csv(f'results/results_file_{dataset_name}_{dt}.csv', index=False)

    friedman_test(all_AUC_results[models[0]], all_AUC_results[models[1]], all_AUC_results[models[2]], dataset_name, dt)


def initial_const_hyperparameters():
    """
    Define the constant hyperparameters for the MixMatch algorithms
    :return: a dictionary containing the hyperparameters
    """
    args = {
        'batch_size': 64,
        'epochs': 200,
        'learning_rate': 0.01,
        'steps': 120,
        'rampup_length': 16,
        'T': 0.5,
        'K': 2,
        'weight_decay': 0.02,
        'ema_decay': 0.999,
    }
    return args


def run_inner_loop(train_val_ds, num_classes, args, y_ds, model_name, image_size):
    """
    Runs the hyperparameters optimizations 3 fold cross validation
    :param train_val_ds: dataset to split to train and validation
    :param num_classes: the amount of classes in the dataset
    :param args: the train hyperparameters
    :param y_ds: dataset labels
    :param model_name: the name of the algorithm to train
    :param image_size: the size of the images
    :return: the mean accuracy of all 3 folds for the given hyperparameters
    """
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    x_placeholder = np.zeros(len(y_ds))
    all_results = []
    for train_indices, val_indices in cv_inner.split(x_placeholder, y_ds):
        if model_name == 'baseline':
            train_samples = []
            val_samples = []
            for samples_idx, samples in enumerate(iter(train_val_ds)):
                if samples_idx in val_indices:
                    val_samples.append(samples)
                elif samples_idx in train_indices:
                    train_samples.append(samples)
            train_samples = list_to_tf_dataset_string(train_samples)
            val_samples = list_to_tf_dataset_string(val_samples)
            train_parsed = process_parsed_dataset(train_samples, num_classes, model_name)
            val_parsed = process_parsed_dataset(val_samples, num_classes, model_name)
            train_parsed = train_parsed.shuffle(buffer_size=100).batch(args['batch_size'])
            val_parsed = val_parsed.shuffle(buffer_size=100).batch(args['batch_size'])
            baseline_model, baseline = train_baseline(num_classes, args, train_parsed, image_size)
            all_metrics = validate_baseline(baseline, val_parsed)
        else:
            train_labeled, train_unlabeled, val_samples = labeled_unlabeled_validation_split(train_indices, val_indices,
                                                                                             train_val_ds)
            train_x = process_parsed_dataset(train_labeled, num_classes, model_name)
            train_u = process_parsed_dataset(train_unlabeled, num_classes, model_name)
            val_samples = process_parsed_dataset(val_samples, num_classes, model_name)
            model, ema_model = train_mixmatch(num_classes, args, train_x, train_u, model_name, val_samples)
            xe_avg, all_metrics = validate(val_samples, ema_model, args, split='val')
        all_results.append(all_metrics['Accuracy'])

    return np.mean(all_results)


def validate(dataset, model, args, split):
    """
    This function performs the inference of the MixMatch algorithm.
    :param dataset: the dataset
    :param model: MixMatch original or improved
    :param args: the train hyperparameters
    :param split: validation or test
    :return: loss and performance metrics
    """
    recall_list = []
    FPR_list = []
    precision_list = []
    AUC_list = []
    AUPRC_list = []
    accuracy = Accuracy()
    xe_avg = Mean()
    dataset = dataset.batch(args['batch_size'])
    for batch in dataset:
        if len(batch['image']) != 64:
            continue
        logits = model(batch['image'], training=False)
        xe_loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch['label'], logits=logits)
        xe_avg(xe_loss)
        pred_probs = tf.nn.softmax(logits=logits)
        y_true = tf.argmax(batch['label'], axis=1, output_type=tf.int32)
        y_pred = tf.argmax(pred_probs, axis=1, output_type=tf.int32)
        accuracy(y_pred, y_true)
        AUC = roc_auc_score(y_true, pred_probs, multi_class='ovr')
        PR = average_precision_score(batch['label'].numpy(), pred_probs.numpy(), average='samples')
        AUPRC_list.append(PR)
        cnf_matrix = confusion_matrix(y_true.numpy(), y_pred.numpy())
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        precision_per_class = []
        recall_per_class = []
        FPR_per_class = []
        for i in range(len(TP)):
            P = (TP[i] + FP[i])
            precision_i = 0 if P == 0 else TP[i] / P
            precision_per_class.append(precision_i)
            TPFN = (TP[i] + FN[i])
            recall_i = 0 if TPFN == 0 else TP[i] / TPFN
            recall_per_class.append(recall_i)
            N = (TN[i] + FN[i])
            FPR_i = 0 if N == 0 else FP[i] / N
            FPR_per_class.append(FPR_i)
        recall_list.append(np.mean(recall_per_class))
        precision_list.append(np.mean(precision_per_class))
        FPR_list.append(np.mean(FPR_per_class))
        AUC_list.append(AUC)
    print(f'{split}- XE Loss: {xe_avg.result():.4f}, {split} Accuracy: {accuracy.result():.3%}')
    all_results = {
        'TPR': np.mean(recall_list),
        'FPR': np.mean(FPR_list),
        'Precision': np.mean(precision_list),
        'AUC': np.mean(AUC_list),
        'AUPRC': np.mean(AUPRC_list),
        'Accuracy': float(accuracy.result())
    }
    return xe_avg, all_results


def train_step(dataset_x, dataset_u, model, ema_model, optimizer, epoch, args, model_name):
    """
    This function performs the train of the MixMatch algorithm for each step. It updates the gradients and accuracy state
    :param dataset_x: labeled images batch
    :param dataset_u: unlabeled images batch
    :param model: MixMatch original or improved model
    :param ema_model: MixMatch original or improved ema model
    :param optimizer: Adam optimized of the model
    :param epoch: the number of epoch
    :param args: the train hyperparameters
    :param model_name: MixMatch original or improved
    :return: train performance metrics
    """
    xe_loss_avg = Mean()
    l2u_loss_avg = Mean()
    total_loss_avg = Mean()
    accuracy = Accuracy()
    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'],
                                                                                    drop_remainder=True)
    iterator_x = iter(shuffle_and_batch(dataset_x))
    iterator_u = iter(shuffle_and_batch(dataset_u))
    progress_bar = tqdm.tqdm(range(args['steps']), unit='batch')
    for batch_num in progress_bar:
        lambda_u = args['lambda_u'] * linear_rampup(epoch + batch_num / args['steps'], args['rampup_length'])
        try:
            batch_x = next(iterator_x)
        except:
            iterator_x = iter(shuffle_and_batch(dataset_x))
            batch_x = next(iterator_x)
        try:
            batch_u = next(iterator_u)
        except:
            iterator_u = iter(shuffle_and_batch(dataset_u))
            batch_u = next(iterator_u)
        args['beta'].assign(np.random.beta(args['alpha'], args['alpha']))
        with tf.GradientTape() as tape:
            # run mixmatch
            gc.collect()
            XU, XUy = mixmatch(model, batch_x['image'], batch_x['label'], batch_u['image'], args['T'], args['K'],
                               args['beta'], model_name)
            logits = [model(XU[0])]
            for batch in XU[1:]:
                logits.append(model(batch))
            logits = interleave(logits, args['batch_size'])
            logits_x = logits[0]
            logits_u = tf.concat(logits[1:], axis=0)
            # compute loss
            xe_loss, l2u_loss = semi_loss(XUy[:args['batch_size']], logits_x, XUy[args['batch_size']:], logits_u)
            total_loss = xe_loss + lambda_u * l2u_loss
        # compute gradients and run optimizer step
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema(model, ema_model, args['ema_decay'])
        weight_decay(model=model, decay_rate=args['weight_decay'] * args['learning_rate'])
        xe_loss_avg(xe_loss)
        l2u_loss_avg(l2u_loss)
        total_loss_avg(total_loss)
        logits_for_accuracy = model(tf.cast(batch_x['image'], dtype=tf.float32), training=False)
        logits_for_accuracy_probs = tf.nn.softmax(logits=logits_for_accuracy)
        accuracy(tf.argmax(batch_x['label'], axis=1, output_type=tf.int32),
                 tf.argmax(logits_for_accuracy_probs, axis=1, output_type=tf.int32))
        progress_bar.set_postfix({
            'Accuracy': f'{accuracy.result():.3%}',
            'XE Loss': f'{xe_loss_avg.result():.4f}',
            'L2U Loss': f'{l2u_loss_avg.result():.4f}',
            'WeightU': f'{lambda_u:.3f}',
            'Total Loss': f'{total_loss_avg.result():.4f}'
        })
    train_metrics = {'accuracy': accuracy, 'xe_loss_avg': xe_loss_avg, 'l2u_loss_avg': l2u_loss_avg,
                     'total_loss_avg': total_loss_avg}

    return train_metrics


def train_mixmatch(num_classes, args, dataset_x, dataset_u, model_name, val_samples):
    """
    This function performs the entire train process of the MixMatch algorithm. For each epoch is calls train step
    :param num_classes: the amount of classes in the dataset
    :param args: the train hyperparameters
    :param dataset_x: labeled images batch
    :param dataset_u: unlabeled images batch
    :param model_name: MixMatch original or improved
    :param val_samples: validation dataset
    :return: trained model and emd model
    """
    model = WideResNet(model_name, num_classes)
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = Adam(args['learning_rate'])
    ema_model = WideResNet(model_name, num_classes)
    ema_model.build(input_shape=(None, 32, 32, 3))
    ema_model.set_weights(model.get_weights())
    args['beta'] = tf.Variable(0., shape=())
    epochs_results = []
    for epoch in range(args['epochs']):
        print(f'epoch: {epoch}')
        train_metrics = train_step(dataset_x, dataset_u, model, ema_model, optimizer, epoch, args, model_name)
        epochs_results.append(train_metrics)
        xe_avg, all_metrics = validate(val_samples, ema_model, args, split='val')
    return model, ema_model


def train_baseline(num_classes, args, x_train, image_size):
    """
    This function performs the train process of the baseline algorithm
    :param num_classes: the amount of classes in the dataset
    :param args: the train hyperparameters
    :param x_train: images batch
    :param image_size: the size of the images in the dataset
    :return: the trained baseline model
    """
    metrics = ['accuracy', Precision(), AUC(), AUC(curve='PR'), TrueNegatives(),
               TruePositives(), FalseNegatives(), FalsePositives(), Recall()]
    baseline = Baseline(image_size)
    baseline_model = baseline.create_baseline(lr=args['learning_rate'], dropout_rate=args['dropout_rate'],
                                                          dense_size=args['dense_size'],
                                                          num_classes=num_classes, metrics=metrics)
    train_history = baseline_model.fit(x_train, epochs=args['epochs'], verbose=2)
    return baseline_model, baseline


def validate_baseline(baseline, x_test):
    """
    This function performs the inference of the baseline model.
    :param baseline: baseline object
    :param x_test: dataset to evaluate
    :return: performance metrics
    """
    loss, all_metrics = baseline.evaluate(x_test)
    return all_metrics


def friedman_test(AUC_0, AUC_1, AUC_2, dataset_name, dt, alpha=0.05):
    """
    This function performs the friedman test for the results of the three algorithms.
    If the test is significant it calls the Nemenyi post-hoc test
    :param AUC_0: AUC results of the original MixMatch for all 10 folds
    :param AUC_1: AUC results of the improved MixMatch for all 10 folds
    :param AUC_2: AUC results of the baseline for all 10 folds
    :param dataset_name: the name of the dataset
    :param dt: date and time
    :param alpha: significance level, 0.05 by default
    """
    with open('results/stats_results.txt', 'a') as results_file:
        stat, pvalue = friedmanchisquare(AUC_0, AUC_1, AUC_2)
        if pvalue <= alpha:
            results_file.write(f'{dataset_name} - {dt} - test is significant, statistic: {stat}, p-value: {pvalue}\n')
            post_hoc_test(AUC_0, AUC_1, AUC_2, dataset_name, dt)
        else:
            results_file.write(f'{dataset_name} - {dt} - test is not significant, statistic: {stat}, p-value: {pvalue}\n')

    results_file.close()


def post_hoc_test(AUC_0, AUC_1, AUC_2, dataset_name, dt):
    """
    This function performs the Nemenyi post-hoc test
    :param AUC_0: AUC results of the original MixMatch for all 10 folds
    :param AUC_1: AUC results of the improved MixMatch for all 10 folds
    :param AUC_2: AUC results of the baseline for all 10 folds
    :param dataset_name: the name of the dataset
    :param dt: date and time
    """
    all_AUC = np.array([AUC_0, AUC_1, AUC_2])
    # perform Nemenyi post-hoc test
    p_values_df = sp.posthoc_nemenyi_friedman(all_AUC.T)
    p_values_df.to_csv(f'results/post_hoc_result_{dataset_name}_{dt}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='', type=str)
    main_args = parser.parse_args()
    main(main_args)
