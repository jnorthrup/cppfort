
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar039-bitwise-not/ar039-bitwise-not_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_bitwise_noti>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe8    	ldr	w8, [sp, #0xc]
10000036c: 2a2803e0    	mvn	w0, w8
100000370: 910043ff    	add	sp, sp, #0x10
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: d10083ff    	sub	sp, sp, #0x20
10000037c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000380: 910043fd    	add	x29, sp, #0x10
100000384: 52800000    	mov	w0, #0x0                ; =0
100000388: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000038c: 97fffff5    	bl	0x100000360 <__Z16test_bitwise_noti>
100000390: 12001c00    	and	w0, w0, #0xff
100000394: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000398: 910083ff    	add	sp, sp, #0x20
10000039c: d65f03c0    	ret
