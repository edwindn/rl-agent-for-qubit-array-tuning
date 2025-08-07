Changes to training:

* train agent to output capacitance matrices and use these to virtualise
* we reward if at the end of the episode the virtualisation is correct
    (in qarray, we just align this with the underlying matrix)