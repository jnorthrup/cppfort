
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf099-state-machine/cf099-state-machine_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_state_machinei>:
100000360: 531f7808    	lsl	w8, w0, #1
100000364: 7100041f    	cmp	w0, #0x1
100000368: 5a9fa100    	csinv	w0, w8, wzr, ge
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800540    	mov	w0, #0x2a               ; =42
100000374: d65f03c0    	ret
