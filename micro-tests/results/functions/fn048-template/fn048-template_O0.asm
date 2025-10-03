
/Users/jim/work/cppfort/micro-tests/results/functions/fn048-template/fn048-template_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <_main>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000458: 52800600    	mov	w0, #0x30               ; =48
10000045c: 9400000a    	bl	0x100000484
100000460: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000464: 910083ff    	add	sp, sp, #0x20
100000468: d65f03c0    	ret

000000010000046c <__Z4funcIiET_S0_>:
10000046c: d10043ff    	sub	sp, sp, #0x10
100000470: b9000fe0    	str	w0, [sp, #0xc]
100000474: b9400fe8    	ldr	w8, [sp, #0xc]
100000478: 1100c100    	add	w0, w8, #0x30
10000047c: 910043ff    	add	sp, sp, #0x10
100000480: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000484 <__stubs>:
100000484: 90000030    	adrp	x16, 0x100004000
100000488: f9400210    	ldr	x16, [x16]
10000048c: d61f0200    	br	x16
