
/Users/jim/work/cppfort/micro-tests/results/functions/fn098-variadic/fn098-variadic_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <_main>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000458: 52800020    	mov	w0, #0x1                ; =1
10000045c: 52800041    	mov	w1, #0x2                ; =2
100000460: 52800c42    	mov	w2, #0x62               ; =98
100000464: 9400000f    	bl	0x1000004a0
100000468: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000046c: 910083ff    	add	sp, sp, #0x20
100000470: d65f03c0    	ret

0000000100000474 <__Z3sumIJiiiEEiDpT_>:
100000474: d10043ff    	sub	sp, sp, #0x10
100000478: b9000fe0    	str	w0, [sp, #0xc]
10000047c: b9000be1    	str	w1, [sp, #0x8]
100000480: b90007e2    	str	w2, [sp, #0x4]
100000484: b9400fe8    	ldr	w8, [sp, #0xc]
100000488: b9400be9    	ldr	w9, [sp, #0x8]
10000048c: b94007ea    	ldr	w10, [sp, #0x4]
100000490: 0b0a0129    	add	w9, w9, w10
100000494: 0b090100    	add	w0, w8, w9
100000498: 910043ff    	add	sp, sp, #0x10
10000049c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004a0 <__stubs>:
1000004a0: 90000030    	adrp	x16, 0x100004000
1000004a4: f9400210    	ldr	x16, [x16]
1000004a8: d61f0200    	br	x16
