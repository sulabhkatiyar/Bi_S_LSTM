import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention_choice
from datasets import *
from utils import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
from eval_val import evaluate
import json
import argparse

# Data parameters
data_folder = 'path_to_data_files'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
dataset='flickr8k'

# Model parameters

emb_dim = 512  # dimension of word embeddings, have to change it to final embeddings size if using ensembles
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 20  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu1, best_bleu2, best_bleu3, best_bleu4 = 0.,0.,0.,0.  # BLEU scores right now
guiding_bleu= 1 # 1: BLEU 1, 2: BLEU-2, 3: BLEU-3, 4: BLEU4 #THE BLEU METRIC USED TO GUIDE THE PROCESS
print_freq = 1000  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None # path to checkpoint, None if none
encoder_dim=4096  


def main():
    global best_bleu1, best_bleu2, best_bleu3, best_bleu4, epochs_since_improvement, checkpoint, start_epoch
    global emb_dim, encoder_dim, data_name, word_map, guiding_bleu, val_loader_single, device

    # Remove previous checkpoints if they exist in same directory 
    if checkpoint == None:
        if 'BEST_checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar' in os.listdir(os.getcwd()):
            os.remove('BEST_checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar')
        if 'checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar' in os.listdir(os.getcwd()):
            os.remove('checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar')


    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    
    if checkpoint is None:
        decoder = DecoderWithAttention_choice(embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       encoder_dim=encoder_dim,
                                       dropout=dropout)

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder_optimizer = None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu_scores'][3]
        best_bleu3 = checkpoint['bleu_scores'][2]
        best_bleu2 = checkpoint['bleu_scores'][1]
        best_bleu1 = checkpoint['bleu_scores'][0]
        decoder = checkpoint['decoder']        
        decoder_optimizer = checkpoint['decoder_optimizer']

        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']


    decoder = decoder.to(device)
    encoder = encoder.to(device)    
    criterion = nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)    

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader_single = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)

    
    # Epochs
    for epoch in range(start_epoch, epochs):

        if epochs_since_improvement == 4:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_learning_rate(decoder_optimizer, 0.95)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.9)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)
        
        recent_bleu1,recent_bleu2,recent_bleu3,recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        if guiding_bleu==4:
            is_best = recent_bleu4 > best_bleu4 
        elif guiding_bleu==3:
            is_best = recent_bleu3 > best_bleu3
        elif guiding_bleu==2:
            is_best = recent_bleu2 > best_bleu2
        elif guiding_bleu==1:
            is_best = recent_bleu1 > best_bleu1
        
        best_bleu1 = max(recent_bleu1, best_bleu1)
        best_bleu2 = max(recent_bleu2, best_bleu2)
        best_bleu3 = max(recent_bleu3, best_bleu3)
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        
        bleu_list=[recent_bleu1,recent_bleu2,recent_bleu3,recent_bleu4]

        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, bleu_list, is_best)

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()  
    encoder.train()

    batch_time = AverageMeter()  
    data_time = AverageMeter()  
    losses = AverageMeter()  
    top5accs = AverageMeter()

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, image_name) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[0][:, 1:]
        scores[0] = pack_padded_sequence(scores[0], decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = criterion(scores[0], targets)
        targets_r = caps_sorted[1][:, 1:]
        scores[1] = pack_padded_sequence(scores[1], decode_lengths, batch_first=True).data
        targets_r = pack_padded_sequence(targets_r, decode_lengths, batch_first=True).data
        loss = loss + criterion(scores[1], targets_r)
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()
        
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores[0], targets, 5) + accuracy(scores[1], targets_r, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()  
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  
    hypotheses = list()  
    hypotheses_r = list() 
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps, image_name) in enumerate(val_loader):

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[0][:, 1:]
            scores_copy = scores[0].clone()
            scores[0] = pack_padded_sequence(scores[0], decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            loss = criterion(scores[0], targets)
            targets_r = caps_sorted[1][:, 1:]
            scores_copy_r = scores[1].clone()
            scores[1] = pack_padded_sequence(scores[1], decode_lengths, batch_first=True).data
            targets_r = pack_padded_sequence(targets_r, decode_lengths, batch_first=True).data
            loss = loss + criterion(scores[1], targets_r)

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores[0], targets, 5) + accuracy(scores[1], targets_r, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            allcaps = allcaps[sort_ind]  
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  
                references.append(img_captions)

            reverse_word_map = {index:word for (word, index) in word_map.items()}

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            _, preds_r = torch.max(scores_copy_r, dim=2)
            preds_r = preds_r.tolist()
            temp_preds_r = list()
            for j, p in enumerate(preds_r):
                temp_preds_r.append(preds_r[j][:decode_lengths[j]])  # remove pads
            preds_r = temp_preds_r
            for j, p in enumerate(preds_r):
                p.reverse()
            hypotheses_r.extend(preds_r)
            assert len(references) == len(hypotheses_r)

        bleu4 = corpus_bleu(references, hypotheses)
        bleu3 = corpus_bleu(references, hypotheses, (1.0/3.0,1.0/3.0,1.0/3.0,))
        bleu2 = corpus_bleu(references, hypotheses, (1.0/2.0,1.0/2.0,))
        bleu1 = corpus_bleu(references, hypotheses, (1.0/1.0,))

        bleu4_r = corpus_bleu(references, hypotheses_r)
        bleu3_r = corpus_bleu(references, hypotheses_r, (1.0/3.0,1.0/3.0,1.0/3.0,))
        bleu2_r = corpus_bleu(references, hypotheses_r, (1.0/2.0,1.0/2.0,))
        bleu1_r = corpus_bleu(references, hypotheses_r, (1.0/1.0,))

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU scores - {bleu1},{bleu2},{bleu3},{bleu4}\n'.format(loss=losses,
                top5=top5accs,
                bleu1=bleu1,bleu2=bleu2,bleu3=bleu3,bleu4=bleu4))

        print(
            'BLEU scores for reverse LSTM - {bleu1_r},{bleu2_r},{bleu3_r},{bleu4_r}\n'.format(bleu1_r=bleu1_r,bleu2_r=bleu2_r,bleu3_r=bleu3_r,bleu4_r=bleu4_r))

        metrics_list = evaluate(val_loader_single, encoder, decoder, criterion, word_map, device)

        # metrics_list[0] has overall model validation performance; metrics_list[1] has forward model performance and metrics_list[2] has backward model performance
        bleu1,bleu2,bleu3,bleu4 = metrics_list[0]
        
    return bleu1,bleu2,bleu3,bleu4


if __name__ == '__main__':
    main()
