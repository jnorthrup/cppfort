
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf090-short-circuit-ternary/cf090-short-circuit-ternary_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z26test_short_circuit_ternaryi>:
100000360: 7100041f    	cmp	w0, #0x1
100000364: 528000a8    	mov	w8, #0x5                ; =5
100000368: 1a9fb500    	csinc	w0, w8, wzr, lt
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800020    	mov	w0, #0x1                ; =1
100000374: d65f03c0    	ret
