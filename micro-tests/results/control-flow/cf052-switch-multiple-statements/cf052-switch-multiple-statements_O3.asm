
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf052-switch-multiple-statements/cf052-switch-multiple-statements_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_switch_multi_stmti>:
100000360: 528000a8    	mov	w8, #0x5                ; =5
100000364: 7100081f    	cmp	w0, #0x2
100000368: 1a9f0108    	csel	w8, w8, wzr, eq
10000036c: 528001a9    	mov	w9, #0xd                ; =13
100000370: 7100041f    	cmp	w0, #0x1
100000374: 1a880120    	csel	w0, w9, w8, eq
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: 528001a0    	mov	w0, #0xd                ; =13
100000380: d65f03c0    	ret
