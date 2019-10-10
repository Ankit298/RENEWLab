/*
 Copyright (c) 2018-2019, Rice University 
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
 Author(s): Peiyao Zhao: pdszpy19930218@163.com 
            Rahman Doost-Mohamamdy: doost@rice.edu
 
----------------------------------------------------------
 Handles received samples from massive-mimo base station 
----------------------------------------------------------
*/

#include "include/receiver.h"
#include "include/macros.h"
#include "include/utils.h"
#include <atomic>
#include <unistd.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

Receiver::Receiver(int n_rx_threads, Config* config, moodycamel::ConcurrentQueue<Event_data>* in_queue)
{
    this->config_ = config;
    radioconfig_ = new RadioConfig(this->config_);

    thread_num_ = n_rx_threads;

    /* initialize random seed: */
    srand(time(NULL));

    message_queue_ = in_queue;
    radioconfig_->radioConfigure();
}

Receiver::~Receiver()
{
    radioconfig_->radioStop();
    delete radioconfig_;
}

std::vector<pthread_t> Receiver::startClientThreads()
{
    std::vector<pthread_t> client_threads;
    if (config_->clPresent) {
        double frameTime = config_->sampsPerSymbol * config_->clFrames[0].size() * 1e3 / config_->rate; // miliseconds
        unsigned frameTimeDelta = (unsigned)(std::ceil(TIME_DELTA / frameTime));
        std::cout << "Frame time delta " << frameTimeDelta << std::endl;

        for (unsigned int i = 0; i < config_->nClSdrs; i++) {
            pthread_t cl_thread_;
            // record the thread id
            dev_profile* profile = new dev_profile;
            profile->tid = i;
            profile->rate = config_->rate;
            profile->nsamps = config_->sampsPerSymbol;
            profile->txSyms = config_->clULSymbols[i].size();
            profile->rxSyms = config_->clDLSymbols[i].size();
            profile->txStartSym = config_->clULSymbols[i].empty() ? 0 : config_->clULSymbols[i][0];
            profile->txFrameDelta = frameTimeDelta;
            profile->radio = &radioconfig_->radios[i];
            profile->core = i + 1 + config_->rx_thread_num + config_->task_thread_num;
            profile->ptr = this;
            // start socket thread
            if (pthread_create(&cl_thread_, NULL, Receiver::clientTxRx_launch, profile) != 0) {
                perror("socket client thread create failed");
                exit(0);
            }
            client_threads.push_back(cl_thread_);
        }
    }

    return client_threads;
}

std::vector<pthread_t> Receiver::startRecvThreads(SampleBuffer* rx_buffer, unsigned in_core_id)
{
    assert(rx_buffer[0].buffer.size() != 0);

    std::vector<pthread_t> created_threads;
    created_threads.resize(thread_num_);
    for (int i = 0; i < thread_num_; i++) {
        // record the thread id
        ReceiverContext* context = new ReceiverContext;
        context->ptr = this;
        context->buffer = rx_buffer;
        context->core_id = in_core_id;
        context->tid = i;
        // start socket thread
        if (pthread_create(&created_threads[i], NULL, Receiver::loopRecv_launch, context) != 0) {
            perror("socket recv thread create failed");
            exit(0);
        }
    }

    sleep(1);
    pthread_cond_broadcast(&cond);
    go();
    return created_threads;
}

void Receiver::completeRecvThreads(const std::vector<pthread_t>& recv_thread)
{
    for (std::vector<pthread_t>::const_iterator it = recv_thread.begin();
         it != recv_thread.end(); ++it)
        pthread_join(*it, NULL);
}

void Receiver::go()
{
    radioconfig_->radioStart(); // hardware trigger
}

void* Receiver::loopRecv_launch(void* in_context)
{
    ReceiverContext* context = (ReceiverContext*)in_context;
    context->ptr->loopRecv(context);
    return 0;
}

void Receiver::loopRecv(ReceiverContext* context)
{
    SampleBuffer* rx_buffer = context->buffer;
    int tid = context->tid;
    int core_id = context->core_id;
    delete context;

    if (config_->core_alloc) {
        printf("pinning thread %d to core %d\n", tid, core_id + tid);
        if (pin_to_core(core_id + tid) != 0) {
            printf("pin thread %d to core %d failed\n", tid, core_id + tid);
            exit(0);
        }
    }

    // Use mutex to sychronize data receiving across threads
    pthread_mutex_lock(&mutex);
    printf("Recv Thread %d: waiting for release\n", tid);

    pthread_cond_wait(&cond, &mutex);
    pthread_mutex_unlock(&mutex); // unlocking for all other threads

    // use token to speed up
    moodycamel::ProducerToken local_ptok(*message_queue_);

    const int bsSdrCh = config_->bsChannel.length();
    int buffer_chunk_size = rx_buffer[0].buffer.size() / config_->getPackageLength();

    // handle two channels at each radio
    // this is assuming buffer_chunk_size is at least 2
    std::atomic_int* pkg_buf_inuse = rx_buffer[tid].pkg_buf_inuse;
    char* buffer = rx_buffer[tid].buffer.data();
    int num_radios = config_->nBsSdrs[0];
    int radio_start = tid * num_radios / thread_num_;
    int radio_end = (tid + 1) * num_radios / thread_num_;

    printf("receiver thread %d has %d radios\n", tid, radio_end - radio_start);

    int cursor = 0;
    while (config_->running) {
        // receive data
        for (int it = radio_start; it < radio_end; it++) {
            Package* pkg[bsSdrCh];
            void* samp[bsSdrCh];
            for (auto ch = 0; ch < bsSdrCh; ++ch) {
                pkg[ch] = (Package*)(buffer + (cursor + ch) * config_->getPackageLength());
                samp[ch] = pkg[ch]->data;
            }
            long long frameTime;
            if (radioconfig_->radioRx(it, samp, frameTime) < 0) {
                config_->running = false;
                break;
            }

            int frame_id = (int)(frameTime >> 32);
            int symbol_id = (int)((frameTime >> 16) & 0xFFFF);
            int ant_id = it * bsSdrCh;
#if DEBUG_PRINT
            for (auto ch = 0; ch < bsSdrCh; ++ch) {
                printf("receive thread %d, frame %d, symbol %d, cell %d, ant %d samples: %d %d %d %d %d %d %d %d ...\n",
                    tid, frame_id, symbol_id, 0, ant_id + ch,
                    pkg[ch]->data[1], pkg[ch]->data[2], pkg[ch]->data[3], pkg[ch]->data[4],
                    pkg[ch]->data[5], pkg[ch]->data[6], pkg[ch]->data[7], pkg[ch]->data[8]);
            }
#endif
            for (auto ch = 0; ch < bsSdrCh; ++ch) {
                // move ptr & set status to full

                int bit = 1 << cursor % sizeof(std::atomic_int);
                int offs = cursor / sizeof(std::atomic_int);
                int old = std::atomic_fetch_or(&pkg_buf_inuse[offs], bit); // now full
                // if buffer was full, exit
                if (old & bit) {
                    printf("thread %d buffer full\n", tid);
                    exit(0);
                }
                // has data, after it is read it should be set to 0
                new (pkg[ch]) Package(frame_id, symbol_id, 0, ant_id + ch);
                // push EVENT_RX_SYMBOL event into the queue
                Event_data package_message;
                package_message.event_type = EVENT_RX_SYMBOL;
                // data records the position of this packet in the buffer & tid of this socket
                // (so that task thread could know which buffer it should visit)
                package_message.data = cursor + ch + tid * buffer_chunk_size;
                if (!message_queue_->enqueue(local_ptok, package_message)) {
                    printf("socket message enqueue failed\n");
                    exit(0);
                }
                cursor++;
                cursor %= buffer_chunk_size;
            }
        }
    }
}

void* Receiver::clientTxRx_launch(void* in_context)
{
    dev_profile* context = (dev_profile*)in_context;
    Receiver* receiver = context->ptr;
    receiver->clientTxRx(context);
    return 0;
}

void Receiver::clientTxRx(dev_profile* context)
{
    struct Radio* radio = context->radio;
    SoapySDR::Device* device = radio->dev;
    SoapySDR::Stream* rxStream = radio->rxs;
    SoapySDR::Stream* txStream = radio->txs;
    int tid = context->tid;
    int txSyms = context->txSyms;
    int rxSyms = context->rxSyms;
    int txStartSym = context->txStartSym;
    unsigned txFrameDelta = context->txFrameDelta;
    int NUM_SAMPS = context->nsamps;

    if (config_->core_alloc) {
        printf("pinning client thread %d to core %d\n", tid, context->core);
        if (pin_to_core(context->core) != 0) {
            printf("pin client thread %d to core %d failed\n", tid, context->core);
            exit(0);
        }
    }

    //while(!d_mutex.try_lock()){}
    //thread_count++;
    //std::cout << "Thread " << tid << ", txSyms " << txSyms << ", rxSyms " << rxSyms << ", txStartSym " << txStartSym << ", rate " << context->rate << ", txFrameDelta " << txFrameDelta << ", nsamps " << NUM_SAMPS << std::endl;
    //d_mutex.unlock();

    delete context;
    std::vector<std::complex<float>> buffs(NUM_SAMPS, 0);
    std::vector<void*> rxbuff(2);
    rxbuff[0] = buffs.data();
    rxbuff[1] = buffs.data();

    std::vector<void*> txbuff(2);
    if (txSyms > 0) {
        txbuff[0] = config_->txdata[tid].data();
        txbuff[1] = config_->txdata[tid].data();
        std::cout << txSyms << " uplink symbols will be sent per frame..." << std::endl;
    }

    int all_trigs = 0;
    struct timespec tv, tv2;
    clock_gettime(CLOCK_MONOTONIC, &tv);

    while (config_->running) {
        clock_gettime(CLOCK_MONOTONIC, &tv2);
        double diff = ((tv2.tv_sec - tv.tv_sec) * 1e9 + (tv2.tv_nsec - tv.tv_nsec)) / 1e9;
        if (diff > 2) {
            int total_trigs = device->readRegister("IRIS30", 92);
            std::cout << "new triggers: " << total_trigs - all_trigs << ", total: " << total_trigs << std::endl;
            all_trigs = total_trigs;
            tv = tv2;
        }
        // receiver loop
        long long rxTime(0);
        long long txTime(0);
        long long firstRxTime(0);
        bool receiveErrors = false;
        for (int i = 0; i < rxSyms; i++) {
            int flags(0);
            int r = device->readStream(rxStream, rxbuff.data(), NUM_SAMPS, flags, rxTime, 1000000);
            if (r == NUM_SAMPS) {
                if (i == 0)
                    firstRxTime = rxTime;
            } else {
                std::cerr << "waiting for receive frames... " << std::endl;
                receiveErrors = true;
                break;
            }
        }
        if (receiveErrors)
            continue; // just go to the next frame

        // transmit loop
        int flags = SOAPY_SDR_HAS_TIME;
        txTime = firstRxTime & 0xFFFFFFFF00000000;
        txTime += ((long long)txFrameDelta << 32);
        txTime += ((long long)txStartSym << 16);
        //printf("rxTime %llx, txTime %llx \n", firstRxTime, txTime);
        if (!config_->running)
            flags |= SOAPY_SDR_END_BURST; //end burst on last iter
        for (int i = 0; i < txSyms; i++) {
            //if (i == txSyms - 1)
            flags |= SOAPY_SDR_END_BURST;
            int r = device->writeStream(txStream, txbuff.data(), NUM_SAMPS, flags, txTime, 1000000);
            if (r == NUM_SAMPS) {
                txTime += 0x10000;
            } else {
                std::cerr << "unexpected writeStream error " << SoapySDR::errToStr(r) << std::endl;
                //goto cleanup;
            }
        }
    }
}
