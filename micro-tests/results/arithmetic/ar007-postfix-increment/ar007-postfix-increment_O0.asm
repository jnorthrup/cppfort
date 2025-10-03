
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar007-postfix-increment/ar007-postfix-increment_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_postfix_inci>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe8    	ldr	w8, [sp, #0xc]
10000036c: 11000509    	add	w9, w8, #0x1
100000370: b9000fe9    	str	w9, [sp, #0xc]
100000374: b9000be8    	str	w8, [sp, #0x8]
100000378: b9400fe8    	ldr	w8, [sp, #0xc]
10000037c: b9400be9    	ldr	w9, [sp, #0x8]
100000380: 0b090100    	add	w0, w8, w9
100000384: 910043ff    	add	sp, sp, #0x10
100000388: d65f03c0    	ret

000000010000038c <_main>:
10000038c: d10083ff    	sub	sp, sp, #0x20
100000390: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000394: 910043fd    	add	x29, sp, #0x10
100000398: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000039c: 528000a0    	mov	w0, #0x5                ; =5
1000003a0: 97fffff0    	bl	0x100000360 <__Z16test_postfix_inci>
1000003a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a8: 910083ff    	add	sp, sp, #0x20
1000003ac: d65f03c0    	ret
