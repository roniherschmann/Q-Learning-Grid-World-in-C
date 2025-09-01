// Q-learning Grid World in C (single-file, no deps)
// Build: gcc -O2 -Wall -Wextra -std=c11 qgrid.c -o qgrid
// Usage examples:
//   ./qgrid --train 10000 --save qtable.bin
//   ./qgrid --load qtable.bin --render --play 3
//   ./qgrid --train 5000 --render --seed 42
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_W 10
#define MAX_H 10
#define ACTIONS 4   // 0=up,1=right,2=down,3=left

typedef struct {
    int w, h;
    int start_x, start_y;
    int goal_x, goal_y;
    int walls[MAX_H][MAX_W];  // 1 if wall
    int step_limit;
    float step_reward;   // typically -1.0
    float goal_reward;   // e.g., +10.0
} Env;

typedef struct {
    int x, y;
} Pos;

typedef struct {
    // Q-table dims: (h*w) x ACTIONS
    int w, h;
    float *q; // size = w*h*ACTIONS
} QModel;

static inline int clamp(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
static inline int state_id(const Env *env, int x, int y){ return y*env->w + x; }
static inline int idxQ(const Env *env, int s, int a){ return s*ACTIONS + a; }

void env_init(Env *env, int w, int h) {
    env->w = w; env->h = h;
    env->start_x = 0; env->start_y = 0;
    env->goal_x = w-1; env->goal_y = h-1;
    memset(env->walls, 0, sizeof(env->walls));
    // Example obstacles for a small maze; tweak as you like
    if (w>=5 && h>=5) {
        env->walls[1][2] = 1;
        env->walls[2][2] = 1;
        env->walls[3][2] = 1;
        env->walls[3][1] = 1;
    }
    env->step_limit = w*h*4;
    env->step_reward = -1.0f;
    env->goal_reward = 10.0f;
}

int env_valid(const Env *env, int x, int y){
    if (x<0 || x>=env->w || y<0 || y>=env->h) return 0;
    if (env->walls[y][x]) return 0;
    return 1;
}

Pos env_step(const Env *env, Pos s, int action, float *reward, int *done){
    Pos ns = s;
    if (action==0) ns.y -= 1;
    else if (action==1) ns.x += 1;
    else if (action==2) ns.y += 1;
    else if (action==3) ns.x -= 1;

    if (!env_valid(env, ns.x, ns.y)) {
        // bump into wall/bounds: stay put, small penalty from step_reward
        ns = s;
    }
    *done = (ns.x==env->goal_x && ns.y==env->goal_y);
    *reward = *done ? env->goal_reward : env->step_reward;
    return ns;
}

void qmodel_alloc(QModel *m, int w, int h) {
    m->w = w; m->h = h;
    m->q = (float*)calloc((size_t)(w*h*ACTIONS), sizeof(float));
    if (!m->q) { fprintf(stderr, "OOM\n"); exit(1); }
}

void qmodel_free(QModel *m) {
    free(m->q); m->q=NULL;
}

int argmax_a(const QModel *m, const Env *env, int s) {
    float best = m->q[idxQ(env, s, 0)];
    int best_a = 0;
    for (int a=1; a<ACTIONS; ++a){
        float v = m->q[idxQ(env, s, a)];
        if (v > best){ best=v; best_a=a; }
    }
    return best_a;
}

float maxQ(const QModel *m, const Env *env, int s){
    float best = m->q[idxQ(env,s,0)];
    for (int a=1; a<ACTIONS; ++a){
        float v = m->q[idxQ(env,s,a)];
        if (v>best) best=v;
    }
    return best;
}

int eps_greedy_action(const QModel *m, const Env *env, int s, float eps){
    // With prob eps choose random action, else greedy
    if (((float)rand() / (float)RAND_MAX) < eps){
        return rand() % ACTIONS;
    } else {
        return argmax_a(m, env, s);
    }
}

void save_qtable(const char *path, const QModel *m){
    FILE *f = fopen(path, "wb");
    if (!f){ perror("fopen"); exit(1); }
    fwrite(&m->w, sizeof(int), 1, f);
    fwrite(&m->h, sizeof(int), 1, f);
    size_t n = (size_t)(m->w*m->h*ACTIONS);
    fwrite(m->q, sizeof(float), n, f);
    fclose(f);
}

int load_qtable(const char *path, QModel *m){
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int w,h;
    if (fread(&w,sizeof(int),1,f)!=1 || fread(&h,sizeof(int),1,f)!=1){
        fclose(f); return 0;
    }
    qmodel_alloc(m, w, h);
    size_t n = (size_t)(w*h*ACTIONS);
    if (fread(m->q, sizeof(float), n, f)!=n){ fclose(f); return 0; }
    fclose(f);
    return 1;
}

void render(const Env *env, Pos agent){
    for (int y=0; y<env->h; ++y){
        for (int x=0; x<env->w; ++x){
            char c='.';
            if (env->walls[y][x]) c='#';
            if (x==env->goal_x && y==env->goal_y) c='G';
            if (x==agent.x && y==agent.y) c='A';
            if (x==env->start_x && y==env->start_y) c = (c=='A')?'A':'S';
            printf("%c ", c);
        }
        printf("\n");
    }
}

void train(Env *env, QModel *m, int episodes, float alpha, float gamma,
           float eps_start, float eps_min, float eps_decay, int render_every){
    double avg_len=0.0, avg_ret=0.0;
    for (int ep=1; ep<=episodes; ++ep){
        // Exponential epsilon decay
        float eps = fmaxf(eps_min, eps_start * expf(-eps_decay * (float)ep));
        Pos s = (Pos){env->start_x, env->start_y};
        int steps=0; float ret=0.0f;
        for (;;){
            if (render_every>0 && (ep%render_every==0)){
                printf("\n[Episode %d | eps=%.3f]\n", ep, eps);
                render(env, s);
            }
            int s_id = state_id(env, s.x, s.y);
            int a = eps_greedy_action(m, env, s_id, eps);

            float r; int done;
            Pos ns = env_step(env, s, a, &r, &done);
            int ns_id = state_id(env, ns.x, ns.y);

            float td_target = r + (done ? 0.0f : gamma * maxQ(m, env, ns_id));
            float *Qsa = &m->q[idxQ(env, s_id, a)];
            *Qsa += alpha * (td_target - *Qsa);

            ret += r;
            s = ns;
            steps++;
            if (done || steps >= env->step_limit) break;
        }
        avg_len += steps;
        avg_ret += ret;
        if (ep % 100 == 0){
            printf("Episode %5d | avg_len: %6.2f | avg_return: %7.3f\n",
                   ep, avg_len/100.0, avg_ret/100.0);
            avg_len = 0.0; avg_ret = 0.0;
        }
    }
}

void play_greedy(const Env *env, const QModel *m, int episodes, int render_flag){
    for (int ep=1; ep<=episodes; ++ep){
        Pos s = (Pos){env->start_x, env->start_y};
        float ret=0; int steps=0;
        printf("\n[Play %d]\n", ep);
        for (;;){
            if (render_flag){
                render(env, s);
                printf("\n");
            }
            int sid = state_id(env, s.x, s.y);
            int a = argmax_a(m, env, sid);
            float r; int done;
            Pos ns = env_step(env, s, a, &r, &done);
            ret += r; steps++;
            s = ns;
            if (done || steps>=env->step_limit) break;
        }
        printf("Return: %.2f | Steps: %d\n", ret, steps);
    }
}

int main(int argc, char **argv){
    // Defaults
    int train_eps = 0;
    int play_eps = 0;
    int render_flag = 0;
    int render_every = 0; // render during training every N episodes (0=off)
    unsigned seed = (unsigned)time(NULL);
    const char *save_path = NULL;
    const char *load_path = NULL;

    // Hyperparams
    int W=5, H=5;
    float alpha=0.1f, gamma=0.99f;
    float eps_start=1.0f, eps_min=0.05f, eps_decay=0.0025f; // tuned for ~10k eps

    // Arg parsing (minimal)
    for (int i=1; i<argc; ++i){
        if (!strcmp(argv[i],"--train") && i+1<argc) train_eps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--play") && i+1<argc) play_eps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--render")) render_flag = 1;
        else if (!strcmp(argv[i],"--render-every") && i+1<argc) render_every = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--save") && i+1<argc) save_path = argv[++i];
        else if (!strcmp(argv[i],"--load") && i+1<argc) load_path = argv[++i];
        else if (!strcmp(argv[i],"--seed") && i+1<argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i],"--size") && i+2<argc){ W=atoi(argv[++i]); H=atoi(argv[++i]); }
        else if (!strcmp(argv[i],"--alpha") && i+1<argc) alpha = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i],"--gamma") && i+1<argc) gamma = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i],"--eps-start") && i+1<argc) eps_start = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i],"--eps-min") && i+1<argc) eps_min = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i],"--eps-decay") && i+1<argc) eps_decay = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i],"--help")){
            printf("Q-learning Grid World\n"
                   "  --train N          Train for N episodes\n"
                   "  --play N           Play greedy policy for N episodes\n"
                   "  --render           Render grid during play\n"
                   "  --render-every N   Render training every N episodes\n"
                   "  --save PATH        Save Q-table to PATH\n"
                   "  --load PATH        Load Q-table from PATH\n"
                   "  --size W H         Grid size (<= %d x %d)\n"
                   "  --alpha A          Learning rate (default 0.1)\n"
                   "  --gamma G          Discount (default 0.99)\n"
                   "  --eps-start E      Epsilon start (default 1.0)\n"
                   "  --eps-min E        Epsilon min (default 0.05)\n"
                   "  --eps-decay D      Epsilon decay (default 0.0025)\n"
                   "  --seed S           RNG seed\n", MAX_W, MAX_H);
            return 0;
        }
    }

    if (W<2 || H<2 || W>MAX_W || H>MAX_H){
        fprintf(stderr, "Invalid --size. Use 2..%dx2..%d\n", MAX_W, MAX_H);
        return 1;
    }
    srand(seed);

    Env env; env_init(&env, W, H);
    QModel q;
    if (load_path){
        if (!load_qtable(load_path, &q)){
            fprintf(stderr, "Failed to load Q-table from %s\n", load_path);
            return 1;
        }
        if (q.w!=W || q.h!=H){
            fprintf(stderr, "Loaded table size %dx%d doesn't match env %dx%d\n",
                    q.w, q.h, W, H);
            qmodel_free(&q);
            return 1;
        }
        printf("Loaded Q-table %dx%d from %s\n", q.w, q.h, load_path);
    } else {
        qmodel_alloc(&q, W, H);
    }

    if (train_eps>0){
        train(&env, &q, train_eps, alpha, gamma, eps_start, eps_min, eps_decay, render_every);
        if (save_path){
            save_qtable(save_path, &q);
            printf("Saved Q-table to %s\n", save_path);
        }
    }
    if (play_eps>0){
        play_greedy(&env, &q, play_eps, render_flag);
    }

    if (train_eps==0 && play_eps==0){
        printf("Nothing to do. Try --train 10000 --save q.bin or --load q.bin --play 5 --render\n");
    }

    qmodel_free(&q);
    return 0;
}