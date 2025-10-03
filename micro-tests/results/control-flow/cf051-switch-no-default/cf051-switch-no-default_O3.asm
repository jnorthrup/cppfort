
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf051-switch-no-default/cf051-switch-no-default_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_switch_no_defaulti>:
100000360: 7100081f    	cmp	w0, #0x2
100000364: 52800288    	mov	w8, #0x14               ; =20
100000368: 5a9f0108    	csinv	w8, w8, wzr, eq
10000036c: 52800149    	mov	w9, #0xa                ; =10
100000370: 7100041f    	cmp	w0, #0x1
100000374: 1a880120    	csel	w0, w9, w8, eq
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: 52800140    	mov	w0, #0xa                ; =10
100000380: d65f03c0    	ret
