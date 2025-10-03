
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf011-simple-ternary/cf011-simple-ternary_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z12test_ternaryi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe9    	ldr	w9, [sp, #0xc]
10000036c: 12800008    	mov	w8, #-0x1               ; =-1
100000370: 71000129    	subs	w9, w9, #0x0
100000374: 1a9fd500    	csinc	w0, w8, wzr, le
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 528000a0    	mov	w0, #0x5                ; =5
100000394: 97fffff3    	bl	0x100000360 <__Z12test_ternaryi>
100000398: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000039c: 910083ff    	add	sp, sp, #0x20
1000003a0: d65f03c0    	ret
