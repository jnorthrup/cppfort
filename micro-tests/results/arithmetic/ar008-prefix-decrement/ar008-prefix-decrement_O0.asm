
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar008-prefix-decrement/ar008-prefix-decrement_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_prefix_deci>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe8    	ldr	w8, [sp, #0xc]
10000036c: 71000500    	subs	w0, w8, #0x1
100000370: b9000fe0    	str	w0, [sp, #0xc]
100000374: 910043ff    	add	sp, sp, #0x10
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: d10083ff    	sub	sp, sp, #0x20
100000380: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000384: 910043fd    	add	x29, sp, #0x10
100000388: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000038c: 528000a0    	mov	w0, #0x5                ; =5
100000390: 97fffff4    	bl	0x100000360 <__Z15test_prefix_deci>
100000394: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000398: 910083ff    	add	sp, sp, #0x20
10000039c: d65f03c0    	ret
