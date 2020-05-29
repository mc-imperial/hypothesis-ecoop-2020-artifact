#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>

int main(int argc, char **argv){
    if (argc != 2) {
        fprintf(stdout, "Usage: callback filename\n");
        return 1;
    }

    char *fifo_commands = getenv("CALLBACKFILESFIFO");
    char *fifo_results = getenv("CALLBACKRESULTSFIFO");

    if ((fifo_commands == NULL) || (fifo_results == NULL)) {
        fprintf(stdout, "Environment variables not set?\n");
        return 2;
    }

    // Sometimes the program gets killed by c-reduce if there's a timeout.
    // This will result in us getting the answer from an old call when we
    // next run. As a result we want to add a uniqueish ID with each call.
    // In theory collisions are possible, but it's so rare that we even need
    // the ID at all that it's not worth worrying about.
    srand(time(0) ^ getpid());
    uint32_t callback = rand();

    char *target_file = realpath(argv[1], NULL);
    assert(target_file != NULL);

    FILE *commands_file = fopen(fifo_commands, "w");
    fprintf(commands_file, "%" PRIu32 " %s\n", callback, target_file);
    fclose(commands_file);
    free(target_file);

    FILE *results_file = fopen(fifo_results, "r");

    while(true){
        unsigned char buf[5];
        int n_read = fread(buf, sizeof(char), 5, results_file);
        if(n_read < 5){
            fprintf(stdout, "Unexpected end of file!\n");
            return 3;
        }
        uint32_t response = 0;
        for(int i = 0; i < 4; i++){
            response <<= 8;
            response |= buf[i];
        };
        if(response == callback){
            return buf[4];
        }
        fprintf(stdout, "Ignoring wrong callback %"PRIu32" != %"PRIu32"\n", callback, response);
        fflush(stdout);
    }
}
