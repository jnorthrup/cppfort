
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf094-tail-recursion/cf094-tail-recursion_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z16factorial_helperii>:
1000003b0: 7100081f    	cmp	w0, #0x2
1000003b4: 5400094b    	b.lt	0x1000004dc <__Z16factorial_helperii+0x12c>
1000003b8: 7100141f    	cmp	w0, #0x5
1000003bc: 54000062    	b.hs	0x1000003c8 <__Z16factorial_helperii+0x18>
1000003c0: aa0003e9    	mov	x9, x0
1000003c4: 14000041    	b	0x1000004c8 <__Z16factorial_helperii+0x118>
1000003c8: 51000408    	sub	w8, w0, #0x1
1000003cc: 9000000a    	adrp	x10, 0x100000000
1000003d0: 7100441f    	cmp	w0, #0x11
1000003d4: 54000082    	b.hs	0x1000003e4 <__Z16factorial_helperii+0x34>
1000003d8: 5280000b    	mov	w11, #0x0               ; =0
1000003dc: aa0003e9    	mov	x9, x0
1000003e0: 14000026    	b	0x100000478 <__Z16factorial_helperii+0xc8>
1000003e4: 121c6d0b    	and	w11, w8, #0xfffffff0
1000003e8: 4f000420    	movi.4s	v0, #0x1
1000003ec: 4f000421    	movi.4s	v1, #0x1
1000003f0: 4e041c21    	mov.s	v1[0], w1
1000003f4: 4e040c02    	dup.4s	v2, w0
1000003f8: 3dc18d43    	ldr	q3, [x10, #0x630]
1000003fc: 4ea38442    	add.4s	v2, v2, v3
100000400: 4b0b0009    	sub	w9, w0, w11
100000404: 6f000463    	mvni.4s	v3, #0x3
100000408: 6f0004e4    	mvni.4s	v4, #0x7
10000040c: 6f000565    	mvni.4s	v5, #0xb
100000410: 6f0005e6    	mvni.4s	v6, #0xf
100000414: aa0b03ec    	mov	x12, x11
100000418: 4f000427    	movi.4s	v7, #0x1
10000041c: 4f000430    	movi.4s	v16, #0x1
100000420: 4ea38451    	add.4s	v17, v2, v3
100000424: 4ea48452    	add.4s	v18, v2, v4
100000428: 4ea58453    	add.4s	v19, v2, v5
10000042c: 4ea29c21    	mul.4s	v1, v1, v2
100000430: 4eb19c00    	mul.4s	v0, v0, v17
100000434: 4eb29ce7    	mul.4s	v7, v7, v18
100000438: 4eb39e10    	mul.4s	v16, v16, v19
10000043c: 4ea68442    	add.4s	v2, v2, v6
100000440: 7100418c    	subs	w12, w12, #0x10
100000444: 54fffee1    	b.ne	0x100000420 <__Z16factorial_helperii+0x70>
100000448: 4ea19c00    	mul.4s	v0, v0, v1
10000044c: 4ea09ce0    	mul.4s	v0, v7, v0
100000450: 4ea09e00    	mul.4s	v0, v16, v0
100000454: 6e004001    	ext.16b	v1, v0, v0, #0x8
100000458: 0ea19c00    	mul.2s	v0, v0, v1
10000045c: 1e26000c    	fmov	w12, s0
100000460: 0e0c3c0d    	mov.s	w13, v0[1]
100000464: 1b0d7d81    	mul	w1, w12, w13
100000468: 6b0b011f    	cmp	w8, w11
10000046c: 54000380    	b.eq	0x1000004dc <__Z16factorial_helperii+0x12c>
100000470: 721e051f    	tst	w8, #0xc
100000474: 540002a0    	b.eq	0x1000004c8 <__Z16factorial_helperii+0x118>
100000478: 121e750c    	and	w12, w8, #0xfffffffc
10000047c: 4e040d21    	dup.4s	v1, w9
100000480: 4b0c0009    	sub	w9, w0, w12
100000484: 4f000420    	movi.4s	v0, #0x1
100000488: 4e041c20    	mov.s	v0[0], w1
10000048c: 3dc18d42    	ldr	q2, [x10, #0x630]
100000490: 4ea28421    	add.4s	v1, v1, v2
100000494: 4b0c016a    	sub	w10, w11, w12
100000498: 6f000462    	mvni.4s	v2, #0x3
10000049c: 4ea19c00    	mul.4s	v0, v0, v1
1000004a0: 4ea28421    	add.4s	v1, v1, v2
1000004a4: 3100114a    	adds	w10, w10, #0x4
1000004a8: 54ffffa1    	b.ne	0x10000049c <__Z16factorial_helperii+0xec>
1000004ac: 6e004001    	ext.16b	v1, v0, v0, #0x8
1000004b0: 0ea19c00    	mul.2s	v0, v0, v1
1000004b4: 0e0c3c0a    	mov.s	w10, v0[1]
1000004b8: 1e26000b    	fmov	w11, s0
1000004bc: 1b0a7d61    	mul	w1, w11, w10
1000004c0: 6b0c011f    	cmp	w8, w12
1000004c4: 540000c0    	b.eq	0x1000004dc <__Z16factorial_helperii+0x12c>
1000004c8: 1b097c21    	mul	w1, w1, w9
1000004cc: 51000528    	sub	w8, w9, #0x1
1000004d0: 7100093f    	cmp	w9, #0x2
1000004d4: aa0803e9    	mov	x9, x8
1000004d8: 54ffff88    	b.hi	0x1000004c8 <__Z16factorial_helperii+0x118>
1000004dc: aa0103e0    	mov	x0, x1
1000004e0: d65f03c0    	ret

00000001000004e4 <__Z9factoriali>:
1000004e4: 7100081f    	cmp	w0, #0x2
1000004e8: 5400006a    	b.ge	0x1000004f4 <__Z9factoriali+0x10>
1000004ec: 52800020    	mov	w0, #0x1                ; =1
1000004f0: d65f03c0    	ret
1000004f4: 7100141f    	cmp	w0, #0x5
1000004f8: 54000082    	b.hs	0x100000508 <__Z9factoriali+0x24>
1000004fc: 52800028    	mov	w8, #0x1                ; =1
100000500: aa0003ea    	mov	x10, x0
100000504: 14000041    	b	0x100000608 <__Z9factoriali+0x124>
100000508: 51000409    	sub	w9, w0, #0x1
10000050c: 9000000b    	adrp	x11, 0x100000000
100000510: 7100441f    	cmp	w0, #0x11
100000514: 540000a2    	b.hs	0x100000528 <__Z9factoriali+0x44>
100000518: 5280000c    	mov	w12, #0x0               ; =0
10000051c: 52800028    	mov	w8, #0x1                ; =1
100000520: aa0003ea    	mov	x10, x0
100000524: 14000025    	b	0x1000005b8 <__Z9factoriali+0xd4>
100000528: 4e040c00    	dup.4s	v0, w0
10000052c: 3dc18d61    	ldr	q1, [x11, #0x630]
100000530: 4ea18400    	add.4s	v0, v0, v1
100000534: 4f000421    	movi.4s	v1, #0x1
100000538: 121c6d2c    	and	w12, w9, #0xfffffff0
10000053c: 6f000462    	mvni.4s	v2, #0x3
100000540: 4b0c000a    	sub	w10, w0, w12
100000544: 6f0004e3    	mvni.4s	v3, #0x7
100000548: 6f000564    	mvni.4s	v4, #0xb
10000054c: 6f0005e5    	mvni.4s	v5, #0xf
100000550: aa0c03e8    	mov	x8, x12
100000554: 4f000426    	movi.4s	v6, #0x1
100000558: 4f000427    	movi.4s	v7, #0x1
10000055c: 4f000430    	movi.4s	v16, #0x1
100000560: 4ea28411    	add.4s	v17, v0, v2
100000564: 4ea38412    	add.4s	v18, v0, v3
100000568: 4ea48413    	add.4s	v19, v0, v4
10000056c: 4ea19c01    	mul.4s	v1, v0, v1
100000570: 4ea69e26    	mul.4s	v6, v17, v6
100000574: 4ea79e47    	mul.4s	v7, v18, v7
100000578: 4eb09e70    	mul.4s	v16, v19, v16
10000057c: 4ea58400    	add.4s	v0, v0, v5
100000580: 71004108    	subs	w8, w8, #0x10
100000584: 54fffee1    	b.ne	0x100000560 <__Z9factoriali+0x7c>
100000588: 4ea19cc0    	mul.4s	v0, v6, v1
10000058c: 4ea09ce0    	mul.4s	v0, v7, v0
100000590: 4ea09e00    	mul.4s	v0, v16, v0
100000594: 6e004001    	ext.16b	v1, v0, v0, #0x8
100000598: 0ea19c00    	mul.2s	v0, v0, v1
10000059c: 1e260008    	fmov	w8, s0
1000005a0: 0e0c3c0d    	mov.s	w13, v0[1]
1000005a4: 1b0d7d08    	mul	w8, w8, w13
1000005a8: 6b0c013f    	cmp	w9, w12
1000005ac: 54000380    	b.eq	0x10000061c <__Z9factoriali+0x138>
1000005b0: 721e053f    	tst	w9, #0xc
1000005b4: 540002a0    	b.eq	0x100000608 <__Z9factoriali+0x124>
1000005b8: 121e752d    	and	w13, w9, #0xfffffffc
1000005bc: 4e040d41    	dup.4s	v1, w10
1000005c0: 4b0d000a    	sub	w10, w0, w13
1000005c4: 4f000420    	movi.4s	v0, #0x1
1000005c8: 4e041d00    	mov.s	v0[0], w8
1000005cc: 3dc18d62    	ldr	q2, [x11, #0x630]
1000005d0: 4ea28421    	add.4s	v1, v1, v2
1000005d4: 4b0d0188    	sub	w8, w12, w13
1000005d8: 6f000462    	mvni.4s	v2, #0x3
1000005dc: 4ea09c20    	mul.4s	v0, v1, v0
1000005e0: 4ea28421    	add.4s	v1, v1, v2
1000005e4: 31001108    	adds	w8, w8, #0x4
1000005e8: 54ffffa1    	b.ne	0x1000005dc <__Z9factoriali+0xf8>
1000005ec: 6e004001    	ext.16b	v1, v0, v0, #0x8
1000005f0: 0ea19c00    	mul.2s	v0, v0, v1
1000005f4: 0e0c3c08    	mov.s	w8, v0[1]
1000005f8: 1e26000b    	fmov	w11, s0
1000005fc: 1b087d68    	mul	w8, w11, w8
100000600: 6b0d013f    	cmp	w9, w13
100000604: 540000c0    	b.eq	0x10000061c <__Z9factoriali+0x138>
100000608: 1b087d48    	mul	w8, w10, w8
10000060c: 51000549    	sub	w9, w10, #0x1
100000610: 7100095f    	cmp	w10, #0x2
100000614: aa0903ea    	mov	x10, x9
100000618: 54ffff88    	b.hi	0x100000608 <__Z9factoriali+0x124>
10000061c: aa0803e0    	mov	x0, x8
100000620: d65f03c0    	ret

0000000100000624 <_main>:
100000624: 52800f00    	mov	w0, #0x78               ; =120
100000628: d65f03c0    	ret
