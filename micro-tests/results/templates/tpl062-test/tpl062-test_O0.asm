
/Users/jim/work/cppfort/micro-tests/results/templates/tpl062-test/tpl062-test_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <_main>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000458: 528007c0    	mov	w0, #0x3e               ; =62
10000045c: 52800021    	mov	w1, #0x1                ; =1
100000460: 9400000c    	bl	0x100000490
100000464: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000468: 910083ff    	add	sp, sp, #0x20
10000046c: d65f03c0    	ret

0000000100000470 <__Z3addIiET_S0_S0_>:
100000470: d10043ff    	sub	sp, sp, #0x10
100000474: b9000fe0    	str	w0, [sp, #0xc]
100000478: b9000be1    	str	w1, [sp, #0x8]
10000047c: b9400fe8    	ldr	w8, [sp, #0xc]
100000480: b9400be9    	ldr	w9, [sp, #0x8]
100000484: 0b090100    	add	w0, w8, w9
100000488: 910043ff    	add	sp, sp, #0x10
10000048c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000490 <__stubs>:
100000490: 90000030    	adrp	x16, 0x100004000
100000494: f9400210    	ldr	x16, [x16]
100000498: d61f0200    	br	x16
