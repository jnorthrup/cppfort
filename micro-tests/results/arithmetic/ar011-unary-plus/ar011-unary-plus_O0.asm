
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar011-unary-plus/ar011-unary-plus_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_unary_plusi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe0    	ldr	w0, [sp, #0xc]
10000036c: 910043ff    	add	sp, sp, #0x10
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: d10083ff    	sub	sp, sp, #0x20
100000378: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000037c: 910043fd    	add	x29, sp, #0x10
100000380: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000384: 528000a0    	mov	w0, #0x5                ; =5
100000388: 97fffff6    	bl	0x100000360 <__Z15test_unary_plusi>
10000038c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000390: 910083ff    	add	sp, sp, #0x20
100000394: d65f03c0    	ret
