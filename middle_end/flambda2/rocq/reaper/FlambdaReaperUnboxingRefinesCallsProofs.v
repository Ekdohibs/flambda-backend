(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* The proofs of FlambdaReaperUnboxingRefinesCallsSpec.v.

   The first statement relates the two rebuilds of the unit, the change
   of calling conventions alone ([cc_rebuild]) and the rebuild with
   unboxing in the mode of the fix ([u_rebuild]:
   [rebuild_unbox_modes_refines_cc]), by the
   relation of FlambdaReaperUnboxingRefinesCallsSimProofs.v ([uc]), a
   simulation. [rebuild_uc] builds it by induction on the unit, the
   continuations in scope being described by [pdesc]s ([cinv]: the masks
   of the first rebuild and the decisions of the second are those of the
   descriptions, but for the return continuation of code that keeps its
   calling convention and the exception continuation of code, which have
   no mask in the first).
   The two rebuilt terms agree but where the second
   unboxes (as in FlambdaReaperUnboxingRefinesProofs.v), where it keeps a
   parameter with no use (a dead variable: [dead] is the variables of
   the unit with no use that are not unboxed), and at calls
   ([call_rebuilt]): the arguments by [arg_ok], group by group
   ([call_args_rel], [call_args_groups]), the code by [code_param_ok]
   ([code_rebuilt]), and the results, returned directly or through a
   wrapper of one rebuild or of both ([uc_result_results]). The wrappers
   are related result by result ([seg_ok]), case by case of the
   decisions of the callee and of the continuation ([result_seg]), and
   the pieces put together ([results_seg]); where the first rebuild
   makes a wrapper, so does the second, whose wrapper passes a result as
   it is only where the first passes it too. The exception continuation
   of a call ([exn_rebuilt]) takes the exception of the callee as a
   parameter takes a result ([exn_param_fits]), and the extra arguments
   are given by the mask of the handler after its first parameter in the
   first rebuild and by its decisions in the second, as the arguments of
   an [Apply_cont] are ([get_extra_args_fst], [get_args_rel2]).
   Code that keeps its calling convention, and its calls, are related
   apart ([code_rebuilt_nc], [call_rebuilt_nc]): both rebuilds give the
   same arguments, poison where the mask of the call does not keep them,
   an unboxed name among them ([args_rel2_mask]), and its result, kept
   with or without a use, is returned as the callee keeps it. A call
   whose code is not known ([call_rebuilt_ind]) is related as one of such
   code, at a shape whose results are those that the continuation takes
   ([ind_results]) and that every code of the unit fits
   ([ind_shape_targets], [ind_fits]); a call that [never_over] says
   over-applies nothing asks no plain result of the code
   ([ind_shape_targets1]), and a call of [f] that over-applies nothing is
   made at a shape that only the code in [reach b f] need fit, [b] being
   whether it is of known arity ([ind_shape_targets_f], [RCHf]); and a call
   that may over-apply at a shape that asks a plain result only of the code
   that [ovr] says may be over-applied ([ind_shape_targets_o2], [OVf]), and
   only the code of [reach false f] and of [rem], which what is left of an
   over-application may call, need fit ([ind_shape_targets_o3], [REMf]). The
   construction is made on the fragment of
   FlambdaReaperUnboxingRefinesReach3Spec.v, for any [reach], [ovr] and
   [rem], which holds that of FlambdaReaperUnboxingRefinesOver2Spec.v
   ([calls_in_over3]), which holds that of
   FlambdaReaperUnboxingRefinesReach2Spec.v ([calls_in_over2]), which holds
   that of FlambdaReaperUnboxingRefinesReachSpec.v ([calls_in_flow2]),
   which holds that of FlambdaReaperUnboxingRefinesOver1Spec.v
   ([calls_in_flow]), which holds that of
   FlambdaReaperUnboxingRefinesOverSpec.v ([calls_in_over1]); the run of
   the first unit must be as [reach], [ovr] and [rem] say
   ([rebuild_unbox_modes_refines_cc_over3]), as it is where [rem] holds
   everywhere ([rebuild_unbox_modes_refines_cc_over2]), where [ovr] does too
   ([rebuild_unbox_modes_refines_cc_reach2]), for a call with
   one group, where [reach false] holds everywhere
   ([rebuild_unbox_modes_refines_cc_reach]), and where [reach] holds
   everywhere ([rebuild_unbox_modes_refines_cc_over1]). A parameter
   kept with no use
   (marked [Keep_unused]) is dead in the relation: the second rebuild
   gives it what the first gives, or poison of its kind. The code that
   the fragment takes to never return ([nr], [NRf]) has its body related
   at [never_ksh] ([code_K_of_body]): its return continuation is not live
   in its body, and the free continuations of the second term are among
   the live ones of the original ([uc_result], [live_conts]), so that the
   return continuation is not free in the second body ([never_ret_dead],
   [uc_never_shadow]). A call of such code that [apply_never_returns] is
   related to the call that the second rebuild makes return to a handler
   [Invalid] ([uc_result_never]: [uc_call_never], [uc_call_never_wraps]).
   The second term then has fewer free continuations than the first, and
   binds fewer statics: a handler that only the first keeps is described
   by [never_ksh] ([uc_let_cont_never]). The statements with no such code
   are those with [nr] empty.
   What the second rebuild keeps of an unboxed allocation, and the reads
   that it keeps of unboxed names, are as in
   FlambdaReaperUnboxingRefinesProofs.v, the closures among the uses of
   an unboxed name adding nothing to them by [check_unboxed_uses]
   ([add_usages_unboxed]). The second statement composes the first with
   FlambdaReaperCallingConventionsProofs.rebuild_cc_refines_observably,
   in its version with local fields ([_local]), of which the statement
   with the old premise is a corollary; the examples are the statements
   applied. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaRenameApartSpec FlambdaKindModel
  FlambdaModel FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesCallsSimProofs
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesReach3Spec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsProofs FlambdaNameAbstractionSpec
  FlambdaReaperPartialGroupsProofs.
Import ListNotations.

(* ======================================================================== *)
(* The shapes of the continuations that are not free                        *)
(* ======================================================================== *)

Scheme uc_mut := Induction for uc Sort Prop
  with uc_handlers_mut := Induction for uc_handlers Sort Prop
  with uc_bindings_mut := Induction for uc_bindings Sort Prop.

Section Agree2.

Variable M : list (node * unboxed_fields).
Variable dead : list variable.
Variable CS : code_id -> option code_shape.
Variable NR : code_id -> bool.
Variable RCH : bool -> variable -> code_id -> Prop.
Variable OV : code_id -> Prop.
Variable REM : code_id -> Prop.

Definition agree_on2 (l : list continuation) (K K' : finite_map kshape)
    : Prop :=
  ∀ k : continuation, k ∈ l -> lookup k K' = lookup k K.

Lemma uc_handlers_keys (K : finite_map kshape)
    (hs hs' : list (continuation * continuation_handler))
    (S : finite_map kshape) :
  uc_handlers M dead CS NR RCH OV REM K hs hs' S -> map fst S = map fst hs.
Proof.
  induction 1 as [ | K k sh ps ps' md md' h h' hs hs' S _ _ _ _ _ IH];
    simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma agree_app2 (S K K' : finite_map kshape) (bound l : list continuation) :
  map fst S = bound -> agree_on2 (without bound l) K K' ->
  agree_on2 l (S ++ K) (S ++ K').
Proof.
  intros HS Hagree k Hk. rewrite !lookup_app.
  destruct (lookup k S) eqn:E; [ reflexivity | ].
  apply Hagree, in_without_iff. split; [ exact Hk | ].
  intros Hin. rewrite <- HS in Hin. exact (lookup_keys k S Hin E).
Qed.

Lemma exn_call_rel_agree (K K' : finite_map kshape) (o : pshape)
    (ap ap' : apply_expr) :
  lookup (ap_exn_continuation ap) K' = lookup (ap_exn_continuation ap) K ->
  exn_call_rel M dead K o ap ap' -> exn_call_rel M dead K' o ap ap'.
Proof. intros H. unfold exn_call_rel. rewrite H. exact (fun x => x). Qed.

(* The shapes of the continuations not free in a term do not matter. *)
Lemma uc_agree :
  ∀ (K : finite_map kshape) (p q : expr),
    uc M dead CS NR RCH OV REM K p q ->
    ∀ K' : finite_map kshape,
      agree_on2 (free_continuations_of_expr p) K K' ->
      uc M dead CS NR RCH OV REM K' p q.
Proof.
  apply (uc_mut M dead CS NR RCH OV REM
           (fun K p q _ =>
              ∀ K', agree_on2 (free_continuations_of_expr p) K K' ->
                    uc M dead CS NR RCH OV REM K' p q)
           (fun K hs hs' S _ =>
              ∀ K',
                agree_on2
                  (flat_map (fun '(_, h) => free_continuations_of_handler h)
                     hs)
                  K K' ->
                uc_handlers M dead CS NR RCH OV REM K' hs hs' S)
           (fun _ _ _ => True)).
  - intros K x kind d body body' Hx Hd Hnh _ IH K' Hagree.
    apply uc_let; [ exact Hx | exact Hd | exact Hnh | apply IH, Hagree ].
  - intros K x d fields l body body' Hf Hxf Hl Hoff _ IH K' Hagree.
    apply uc_alloc with (fields := fields); try assumption.
    apply IH, Hagree.
  - intros K x y fields fields_y ls body body' Hf Hxf Hy Hne Hl _ IH K'
      Hagree.
    apply (uc_alias M dead CS NR RCH OV REM K' x y fields fields_y ls);
      try assumption.
    apply IH, Hagree.
  - intros K x kind d f s y fields fields_y nested l body body' Hf Hxf Hr Hs
      Hy Hne Hin Hl _ IH K' Hagree.
    eapply uc_load; try eassumption. apply IH, Hagree.
  - intros K x kind d f s y v fields body body' Hx Hr Hs Hf Hin _ IH K'
      Hagree.
    eapply uc_read; try eassumption. apply IH, Hagree.
  - intros K bindings bindings' body body' Hb _ _ IH K' Hagree.
    apply uc_static; [ exact Hb | ]. apply IH.
    intros k Hk. apply Hagree. simpl. apply in_or_app. right. exact Hk.
  - intros K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' _ IHh _ IHb
      Hsk' K' Hagree.
    apply (uc_let_cont M dead CS NR RCH OV REM K' k sh ps ps' md md' h h'
             body body');
      try assumption.
    + apply IHh. intros k' Hk. apply Hagree. simpl. apply in_or_app.
      left. exact Hk.
    + apply IHb. intros k' Hk. rewrite !lookup_cons.
      destruct (Pos.eqb k' k) eqn:E; [ reflexivity | ].
      apply Hagree. simpl. apply in_or_app. right. apply in_without_iff.
      split; [ exact Hk | ]. intros [Heq | []]. subst k'.
      rewrite Pos.eqb_refl in E. discriminate.
  - intros K S hs hs' body body' Hhs IHhs Hst _ IHb K' Hagree.
    pose proof (uc_handlers_keys _ _ _ _ Hhs) as Hkeys.
    apply (uc_let_cont_rec M dead CS NR RCH OV REM K' S);
      [ apply IHhs | exact Hst | apply IHb ];
      apply (agree_app2 S K K' (map fst hs)); try exact Hkeys;
      intros k Hk; apply Hagree; simpl; apply in_without_iff;
      apply in_without_iff in Hk as [Hk Hnot]; (split; [ | exact Hnot ]);
      apply in_or_app; [ left | right ]; exact Hk.
  - intros K ac ac' (Hk & Htrap & Hargs & Hnv) K' Hagree.
    apply uc_apply_cont. split; [ exact Hk | split; [ exact Htrap | ] ].
    unfold kshape_at.
    rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)).
    exact (conj Hargs Hnv).
  - intros K scrutinee arms arms' Hs Harms K' Hagree.
    apply uc_switch; [ exact Hs | ]. simpl in Hagree. clear Hs.
    induction Harms as [ | [d ac] [d' ac'] arms arms'
                           [Hd (Hk & Htrap & Hargs & Hnv)] _ IH];
      constructor.
    + split; [ exact Hd | ]. split; [ exact Hk | split; [ exact Htrap | ] ].
      simpl in Hk, Htrap, Hargs, Hnv |- *. unfold kshape_at in Hargs |- *.
      rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)).
      exact (conj Hargs Hnv).
    + apply IH. intros k Hin. apply Hagree. simpl. right. apply in_or_app.
      right. exact Hin.
  - intros K K' _. apply uc_invalid.
  - intros K ap ap' cs k sh Hcall Hexn Hk Hk' Hlk Ht K' Hagree.
    apply (uc_call M dead CS NR RCH OV REM K' ap ap' cs k sh); try assumption.
    + apply (exn_call_rel_agree K); [ | exact Hexn ].
      apply Hagree. simpl. right. left. reflexivity.
    + rewrite <- Hlk. apply Hagree. left. exact Hk.
  - intros K ap ap' cs k sh w wps wmd wargs Hcall Hexn Hk Hw Hwe Hlk Hok K'
      Hagree.
    apply (uc_call_wrap M dead CS NR RCH OV REM K' ap ap' cs k sh w wps wmd
             wargs);
      try assumption.
    + apply (exn_call_rel_agree K); [ | exact Hexn ].
      apply Hagree. simpl. right. left. reflexivity.
    + rewrite <- Hlk. apply Hagree. left. exact Hk.
  - intros K ap ap' cs k sh KS wc cps cmd cargs w wps wmd wargs Hcall Hexn
      Hwc Hw Hwce Hwe Hlk Hcmd Hok K' Hagree.
    apply (uc_call_wraps M dead CS NR RCH OV REM K' ap ap' cs k sh KS wc cps
             cmd cargs w wps wmd wargs); try assumption.
    + apply (exn_call_rel_agree K); [ | exact Hexn ].
      apply Hagree.
      change (ap_exn_continuation ap
              ∈ apply_cont_continuations
                  {| ac_continuation := k; ac_args := cargs;
                     ac_trap_action := None |}
                ++ without [wc] [ap_continuation ap; ap_exn_continuation ap]).
      apply in_or_app. right.
      apply in_without_iff. split; [ right; left; reflexivity | ].
      intros [Heq | []]. exact (Hwce Heq).
    + rewrite <- Hlk. apply Hagree. left. reflexivity.
  - intros K ap ap' cs id w wps wmd Hcall Hexn Hkind Hcs Hnr Hw Hwe K'
      Hagree.
    apply (uc_call_never M dead CS NR RCH OV REM K' ap ap' cs id w wps wmd);
      try assumption.
    apply (exn_call_rel_agree K); [ | exact Hexn ].
    apply Hagree. simpl. right. left. reflexivity.
  - intros K ap ap' cs id wc cps cmd k cargs w wps wmd Hcall Hexn Hkind Hcs
      Hnr Hwc Hw Hwce Hwe K' Hagree.
    apply (uc_call_never_wraps M dead CS NR RCH OV REM K' ap ap' cs id wc
             cps cmd k cargs w wps wmd); try assumption.
    apply (exn_call_rel_agree K); [ | exact Hexn ].
    apply Hagree.
    change (ap_exn_continuation ap
            ∈ apply_cont_continuations
                {| ac_continuation := k; ac_args := cargs;
                   ac_trap_action := None |}
              ++ without [wc] [ap_continuation ap; ap_exn_continuation ap]).
    apply in_or_app. right.
    apply in_without_iff. split; [ right; left; reflexivity | ].
    intros [Heq | []]. exact (Hwce Heq).
  - intros K k h body body' _ IHb K' Hagree.
    apply uc_let_cont_never. apply IHb.
    intros k' Hk. rewrite !lookup_cons.
    destruct (Pos.eqb k' k) eqn:E; [ reflexivity | ].
    apply Hagree. simpl. apply in_or_app. right. apply in_without_iff.
    split; [ exact Hk | ]. intros [Heq | []]. subst k'.
    rewrite Pos.eqb_refl in E. discriminate.
  - intros K K' _. apply uc_handlers_nil.
  - intros K k sh ps ps' md md' h h' hs hs' S Hps Hmd Hmd' _ IHh _ IHhs Hsk' K'
      Hagree.
    apply uc_handlers_cons; try assumption.
    + apply IHh. intros k' Hk. apply Hagree. simpl. apply in_or_app. left.
      exact Hk.
    + apply IHhs. intros k' Hk. apply Hagree. simpl. apply in_or_app.
      right. exact Hk.
  - exact I.
  - intros; exact I.
  - intros; exact I.
  - intros; exact I.
Qed.

Lemma uc_handlers_keys2 (K : finite_map kshape)
    (hs hs' : list (continuation * continuation_handler))
    (S : finite_map kshape) :
  uc_handlers M dead CS NR RCH OV REM K hs hs' S -> map fst S = map fst hs'.
Proof.
  induction 1 as [ | K k sh ps ps' md md' h h' hs hs' S _ _ _ _ _ IH];
    simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

(* [K'] is [K] but at continuations not in [l], which it may describe by
   [never_ksh]. *)
Definition never_ext (l : list continuation) (K K' : finite_map kshape)
    : Prop :=
  ∀ k : continuation,
    lookup k K' = lookup k K ∨ (k ∉ l ∧ lookup k K' = Some never_ksh).

Lemma never_ext_incl (l l' : list continuation) (K K' : finite_map kshape) :
  incl l l' -> never_ext l' K K' -> never_ext l K K'.
Proof.
  intros Hincl H k. destruct (H k) as [Heq | [Hn Hl]]; [ left; exact Heq | ].
  right. split; [ intros Hk; exact (Hn (Hincl k Hk)) | exact Hl ].
Qed.

Lemma never_ext_same (l : list continuation) (K K' : finite_map kshape)
    (k : continuation) :
  never_ext l K K' -> k ∈ l -> lookup k K' = lookup k K.
Proof.
  intros H Hk. destruct (H k) as [Heq | [Hn _]]; [ exact Heq | ].
  exfalso. exact (Hn Hk).
Qed.

Lemma never_ext_app (S K K' : finite_map kshape)
    (bound l : list continuation) :
  map fst S = bound -> never_ext (without bound l) K K' ->
  never_ext l (S ++ K) (S ++ K').
Proof.
  intros HS H k. rewrite !lookup_app.
  destruct (lookup k S) eqn:E; [ left; reflexivity | ].
  destruct (H k) as [Heq | [Hn Hl]]; [ left; exact Heq | ].
  right. split; [ | exact Hl ].
  intros Hk. apply Hn, in_without_iff. split; [ exact Hk | ].
  intros Hin. rewrite <- HS in Hin. exact (lookup_keys k S Hin E).
Qed.

(* A continuation that the second term does not apply may be described by
   [never_ksh]: every case of the relation that reads a continuation in
   [K] has it free in the second term. *)
Lemma uc_never_ext :
  ∀ (K : finite_map kshape) (p q : expr),
    uc M dead CS NR RCH OV REM K p q ->
    ∀ K' : finite_map kshape,
      never_ext (free_continuations_of_expr q) K K' ->
      uc M dead CS NR RCH OV REM K' p q.
Proof.
  apply (uc_mut M dead CS NR RCH OV REM
           (fun K p q _ =>
              ∀ K', never_ext (free_continuations_of_expr q) K K' ->
                    uc M dead CS NR RCH OV REM K' p q)
           (fun K hs hs' S _ =>
              ∀ K',
                never_ext
                  (flat_map (fun '(_, h) => free_continuations_of_handler h)
                     hs')
                  K K' ->
                uc_handlers M dead CS NR RCH OV REM K' hs hs' S)
           (fun _ _ _ => True)).
  - intros K x kind d body body' Hx Hd Hnh _ IH K' H.
    apply uc_let; [ exact Hx | exact Hd | exact Hnh | apply IH, H ].
  - intros K x d fields l body body' Hf Hxf Hl Hoff _ IH K' H.
    apply uc_alloc with (fields := fields); try assumption.
    apply IH. rewrite (proj1 (lets_free_continuations _ body')) in H.
    exact H.
  - intros K x y fields fields_y ls body body' Hf Hxf Hy Hne Hl _ IH K' H.
    apply (uc_alias M dead CS NR RCH OV REM K' x y fields fields_y ls);
      try assumption.
    apply IH. rewrite (proj1 (lets_free_continuations _ body')) in H.
    exact H.
  - intros K x kind d f s y fields fields_y nested l body body' Hf Hxf Hr Hs
      Hy Hne Hin Hl _ IH K' H.
    eapply uc_load; try eassumption.
    apply IH. rewrite (proj1 (lets_free_continuations _ body')) in H.
    exact H.
  - intros K x kind d f s y v fields body body' Hx Hr Hs Hf Hin _ IH K' H.
    eapply uc_read; try eassumption. apply IH, H.
  - intros K bindings bindings' body body' Hb _ _ IH K' H.
    apply uc_static; [ exact Hb | ]. apply IH.
    eapply never_ext_incl; [ | exact H ]. intros k Hk.
    cbn [free_continuations_of_expr]. apply in_or_app. right. exact Hk.
  - intros K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' _ IHh _ IHb
      Hsk' K' H.
    apply (uc_let_cont M dead CS NR RCH OV REM K' k sh ps ps' md md' h h'
             body body');
      try assumption.
    + apply IHh. eapply never_ext_incl; [ | exact H ]. intros k' Hk.
      cbn [free_continuations_of_expr free_continuations_of_handler].
      apply in_or_app. left. exact Hk.
    + apply IHb.
      apply (never_ext_app [(k, Kshape sh (map snd ps) true)] K K' [k] _
               eq_refl).
      eapply never_ext_incl; [ | exact H ]. intros k' Hk.
      cbn [free_continuations_of_expr free_continuations_of_handler].
      apply in_or_app. right. exact Hk.
  - intros K S hs hs' body body' Hhs IHhs Hst _ IHb K' H.
    pose proof (uc_handlers_keys2 _ _ _ _ Hhs) as Hkeys.
    apply (uc_let_cont_rec M dead CS NR RCH OV REM K' S);
      [ apply IHhs | exact Hst | apply IHb ];
      apply (never_ext_app S K K' (map fst hs')); try exact Hkeys;
      eapply never_ext_incl; try exact H; intros k Hk;
      cbn [free_continuations_of_expr]; apply in_without_iff;
      apply in_without_iff in Hk as [Hk Hnot]; (split; [ | exact Hnot ]);
      apply in_or_app; [ left | right ]; exact Hk.
  - intros K ac ac' (Hk & Htrap & Hargs & Hnv) K' H.
    assert (Heq : lookup (ac_continuation ac) K'
                  = lookup (ac_continuation ac) K).
    { apply (never_ext_same _ _ _ _ H).
      cbn [free_continuations_of_expr apply_cont_continuations].
      left. exact Hk. }
    apply uc_apply_cont. split; [ exact Hk | split; [ exact Htrap | ] ].
    unfold kshape_at. rewrite Heq. exact (conj Hargs Hnv).
  - intros K scrutinee arms arms' Hs Harms K' H.
    apply uc_switch; [ exact Hs | ]. cbn [free_continuations_of_expr] in H.
    clear Hs.
    induction Harms as [ | [d ac] [d' ac'] arms arms'
                           [Hd (Hk & Htrap & Hargs & Hnv)] _ IH];
      constructor.
    + split; [ exact Hd | ]. split; [ exact Hk | split; [ exact Htrap | ] ].
      simpl in Hk, Htrap, Hargs, Hnv |- *. unfold kshape_at in Hargs |- *.
      rewrite (never_ext_same _ _ _ (ac_continuation ac) H);
        [ exact (conj Hargs Hnv) | ].
      cbn [flat_map]. apply in_or_app. left.
      cbn [snd apply_cont_continuations]. left. exact Hk.
    + apply IH. eapply never_ext_incl; [ | exact H ]. intros k Hin.
      cbn [flat_map]. apply in_or_app. right. exact Hin.
  - intros K K' _. apply uc_invalid.
  - intros K ap ap' cs k sh Hcall Hexn Hk Hk' Hlk Ht K' H.
    pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He).
    apply (uc_call M dead CS NR RCH OV REM K' ap ap' cs k sh); try assumption.
    + apply (exn_call_rel_agree K); [ | exact Hexn ].
      apply (never_ext_same _ _ _ _ H). right. left. exact He.
    + rewrite <- Hlk. apply (never_ext_same _ _ _ _ H). left. exact Hk'.
  - intros K ap ap' cs k sh w wps wmd wargs Hcall Hexn Hk Hw Hwe Hlk Hok K'
      H.
    pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He).
    apply (uc_call_wrap M dead CS NR RCH OV REM K' ap ap' cs k sh w wps wmd
             wargs);
      try assumption.
    + apply (exn_call_rel_agree K); [ | exact Hexn ].
      apply (never_ext_same _ _ _ _ H). unfold wrap.
      cbn [free_continuations_of_expr free_continuations_of_handler
           apply_cont_continuations ac_continuation ac_trap_action].
      apply in_or_app. right. apply in_without_iff.
      split; [ right; left; exact He | ].
      intros [Heq | []]. exact (Hwe Heq).
    + rewrite <- Hlk. apply (never_ext_same _ _ _ _ H). unfold wrap.
      cbn [free_continuations_of_expr free_continuations_of_handler
           apply_cont_continuations ac_continuation ac_trap_action].
      apply in_or_app. left. left. reflexivity.
  - intros K ap ap' cs k sh KS wc cps cmd cargs w wps wmd wargs Hcall Hexn
      Hwc Hw Hwce Hwe Hlk Hcmd Hok K' H.
    pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He).
    apply (uc_call_wraps M dead CS NR RCH OV REM K' ap ap' cs k sh KS wc cps
             cmd cargs w wps wmd wargs); try assumption.
    + apply (exn_call_rel_agree K); [ | exact Hexn ].
      apply (never_ext_same _ _ _ _ H). unfold wrap.
      cbn [free_continuations_of_expr free_continuations_of_handler
           apply_cont_continuations ac_continuation ac_trap_action].
      apply in_or_app. right. apply in_without_iff.
      split; [ right; left; exact He | ].
      intros [Heq | []]. exact (Hwe Heq).
    + rewrite <- Hlk. apply (never_ext_same _ _ _ _ H). unfold wrap.
      cbn [free_continuations_of_expr free_continuations_of_handler
           apply_cont_continuations ac_continuation ac_trap_action].
      apply in_or_app. left. left. reflexivity.
  - intros K ap ap' cs id w wps wmd Hcall Hexn Hkind Hcs Hnr Hw Hwe K' H.
    pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He).
    apply (uc_call_never M dead CS NR RCH OV REM K' ap ap' cs id w wps wmd);
      try assumption.
    apply (exn_call_rel_agree K); [ | exact Hexn ].
    apply (never_ext_same _ _ _ _ H).
    cbn [free_continuations_of_expr free_continuations_of_handler app].
    apply in_without_iff. split; [ right; left; exact He | ].
    intros [Heq | []]. exact (Hwe Heq).
  - intros K ap ap' cs id wc cps cmd k cargs w wps wmd Hcall Hexn Hkind Hcs
      Hnr Hwc Hw Hwce Hwe K' H.
    pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He).
    apply (uc_call_never_wraps M dead CS NR RCH OV REM K' ap ap' cs id wc
             cps cmd k cargs w wps wmd); try assumption.
    apply (exn_call_rel_agree K); [ | exact Hexn ].
    apply (never_ext_same _ _ _ _ H).
    cbn [free_continuations_of_expr free_continuations_of_handler app].
    apply in_without_iff. split; [ right; left; exact He | ].
    intros [Heq | []]. exact (Hwe Heq).
  - intros K k h body body' _ IHb K' H.
    apply uc_let_cont_never. apply IHb.
    apply (never_ext_app [(k, never_ksh)] K K' [k] _ eq_refl).
    eapply never_ext_incl; [ | exact H ]. intros k' Hk.
    apply in_without_iff in Hk as [Hk _]. exact Hk.
  - intros K K' _. apply uc_handlers_nil.
  - intros K k sh ps ps' md md' h h' hs hs' S Hps Hmd Hmd' _ IHh _ IHhs Hsk' K'
      H.
    apply uc_handlers_cons; try assumption.
    + apply IHh. eapply never_ext_incl; [ | exact H ]. intros k' Hk.
      simpl. apply in_or_app. left. exact Hk.
    + apply IHhs. eapply never_ext_incl; [ | exact H ]. intros k' Hk.
      simpl. apply in_or_app. right. exact Hk.
  - exact I.
  - intros; exact I.
  - intros; exact I.
  - intros; exact I.
Qed.

(* The continuation [k], not free in the second term, may be described by
   [never_ksh]. *)
Lemma uc_never (K : finite_map kshape) (p q : expr) (k : continuation) :
  uc M dead CS NR RCH OV REM K p q -> k ∉ free_continuations_of_expr q ->
  uc M dead CS NR RCH OV REM ((k, never_ksh) :: K) p q.
Proof.
  intros Huc Hk. apply (uc_never_ext K p q Huc). intros k0.
  rewrite lookup_cons. destruct (Pos.eqb k0 k) eqn:E; [ | left; reflexivity ].
  apply Pos.eqb_eq in E. subst k0. right. split; [ exact Hk | reflexivity ].
Qed.

(* The same, [k] described by another shape before. *)
Lemma uc_never_shadow (K : finite_map kshape) (p q : expr) (k : continuation)
    (ksh : kshape) :
  uc M dead CS NR RCH OV REM ((k, ksh) :: K) p q ->
  k ∉ free_continuations_of_expr q ->
  uc M dead CS NR RCH OV REM ((k, never_ksh) :: K) p q.
Proof.
  intros Huc Hk.
  apply (uc_agree _ _ _ (uc_never _ p q k Huc Hk)).
  intros k' _. rewrite !lookup_cons. destruct (Pos.eqb k' k); reflexivity.
Qed.

End Agree2.

(* ======================================================================== *)
(* The two rebuilds of a unit of the fragment                              *)
(* ======================================================================== *)

Definition pshape_of (p : pdesc) : list pshape :=
  if pd_keep p
  then
    [match pd_decision p with
     | Unbox fields => Psh_unboxed fields
     | Keep_unused kind => Psh_dead kind
     | Decided _ => Psh_plain
     end]
  else [].

Definition kept_kind (p : pdesc) : list flambda_kind :=
  if pd_keep p then [pd_kind p] else [].

Definition K_of (C : finite_map (list pdesc * bool)) : finite_map kshape :=
  map
    (fun '(k, (ps, h)) =>
       (k, Kshape (flat_map pshape_of ps) (flat_map kept_kind ps) h))
    C.

Lemma lookup_K_of (C : finite_map (list pdesc * bool)) (k : continuation) :
  lookup k (K_of C)
  = option_map
      (fun e => Kshape (flat_map pshape_of (fst e))
                  (flat_map kept_kind (fst e)) (snd e))
      (lookup k C).
Proof.
  induction C as [ | [k' [ps h]] C IH]; [ reflexivity | ]. unfold K_of.
  cbn [map]. rewrite !lookup_cons. destruct (Pos.eqb k k');
    [ reflexivity | exact IH ].
Qed.

(* A continuation described by pdescs is not [never_ksh]: its kinds are as
   many as its shapes. *)
Lemma K_of_not_never (C : finite_map (list pdesc * bool)) (k : continuation) :
  lookup k (K_of C) <> Some never_ksh.
Proof.
  rewrite lookup_K_of. destruct (lookup k C) as [[ps h] | ];
    cbn [option_map fst snd]; [ | discriminate ].
  intros Heq. injection Heq as Hsh Hks _.
  assert (Hlen : length (flat_map pshape_of ps)
                 = length (flat_map kept_kind ps)).
  { clear. induction ps as [ | p ps IH]; [ reflexivity | ].
    cbn [flat_map]. rewrite !length_app, IH. unfold pshape_of, kept_kind.
    destruct (pd_keep p); reflexivity. }
  rewrite Hsh, Hks in Hlen. discriminate Hlen.
Qed.


(* ======================================================================== *)
(* Paths, renamings and pairs                                               *)
(* ======================================================================== *)

Lemma fields_list_eqb_eq (p p' : list field) :
  fields_list_eqb p p' = true -> p = p'.
Proof.
  revert p'. induction p as [ | f p IH]; intros [ | f' p'] H; simpl in H;
    try discriminate H; [ reflexivity | ].
  apply andb_true_iff in H as [Hf H]. apply field_eqb_eq in Hf. subst f'.
  rewrite (IH p' H). reflexivity.
Qed.

Lemma paths_eqb_eq (l l' : list (list field * flambda_kind)) :
  paths_eqb l l' = true -> l = l'.
Proof.
  revert l'. induction l as [ | [p k] l IH]; intros [ | [p' k'] l'] H;
    simpl in H; try discriminate H; [ reflexivity | ].
  apply andb_true_iff in H as [H Hl]. apply andb_true_iff in H as [Hp Hk].
  apply fields_list_eqb_eq in Hp. apply kind_eqb_eq in Hk. subst p' k'.
  rewrite (IH l' Hl). reflexivity.
Qed.

Lemma no_leaves_nil (fields : unboxed_fields) :
  no_leaves fields = true -> fields_leaves fields = [].
Proof.
  unfold no_leaves. destruct (fields_leaves fields); [ reflexivity | ].
  discriminate.
Qed.

Lemma no_leaves_paths (fields : unboxed_fields) :
  no_leaves fields = true -> fields_paths fields = [].
Proof.
  intros H. apply no_leaves_nil in H.
  pose proof (Forall2_length (fields_leaves_paths fields)) as Hl.
  rewrite H in Hl. destruct (fields_paths fields); [ reflexivity | ].
  discriminate Hl.
Qed.

(* The pairs of fields with no variable are none. *)
Lemma pairs_no_leaves (fields fields' : unboxed_fields)
    (pairs : list (variable * variable * flambda_kind)) :
  unboxed_field_pairs Value (Unboxed fields) (Unboxed fields') = Some pairs ->
  fields_leaves fields = [] -> pairs = [].
Proof.
  intros Hp Hl.
  destruct (pairs_spec (Unboxed fields) Value (Unboxed fields') pairs Hp)
    as (_ & _ & Hc & _).
  change (unboxed_field_leaves Value (Unboxed fields)) with
    (fields_leaves fields) in Hc.
  rewrite Hl in Hc. destruct pairs; [ reflexivity | discriminate Hc ].
Qed.

Lemma rename_unboxed_cons (f : field) (u : unboxed_field)
    (rest : unboxed_fields) (n : positive) :
  rename_unboxed_field (Unboxed ((f, u) :: rest)) n
  = let '(u', n1) := rename_unboxed_field u n in
    match rename_unboxed_field (Unboxed rest) n1 with
    | (Unboxed rest', n2) => (Unboxed ((f, u') :: rest'), n2)
    | (Not_unboxed _, n2) => (Not_unboxed n2, n2)
    end.
Proof.
  simpl. destruct (rename_unboxed_field u n) as [u' n1]. simpl.
  match goal with
  | |- context [?F rest n1] => destruct (F rest n1)
  end.
  reflexivity.
Qed.

(* A renaming keeps the paths, and gives the leaves names in a range,
   each once. *)
Lemma rename_ok (u : unboxed_field) :
  ∀ (k : flambda_kind) (n : positive),
    (∃ fields', ∀ fields, u = Unboxed fields ->
       fst (rename_unboxed_field u n) = Unboxed fields')
    ∧ ufield_paths k (fst (rename_unboxed_field u n)) = ufield_paths k u
    ∧ NoDup (map fst (unboxed_field_leaves k (fst (rename_unboxed_field u n))))
    ∧ (∀ x : variable,
         x ∈ map fst (unboxed_field_leaves k (fst (rename_unboxed_field u n)))
         -> (n <= x)%positive ∧ (x < snd (rename_unboxed_field u n))%positive)
    ∧ (n <= snd (rename_unboxed_field u n))%positive.
Proof.
  induction u as [v | fields IH] using unboxed_field_rec; intros k n.
  - simpl. split; [ exists []; intros fields H; discriminate H | ].
    split; [ reflexivity | ]. split; [ repeat constructor; intros [] | ].
    split; [ intros x [<- | []]; lia | lia ].
  - revert IH n.
    induction fields as [ | [f0 u0] rest IHrest]; intros IH n.
    + simpl. split; [ exists []; reflexivity | ].
      split; [ reflexivity | ]. split; [ constructor | ].
      split; [ intros x [] | lia ].
    + rewrite rename_unboxed_cons.
      destruct (rename_unboxed_field u0 n) as [u0' n1] eqn:Hu0.
      destruct (IH f0 u0 (or_introl eq_refl) (field_kind f0) n)
        as (_ & Hp0 & Hnd0 & Hr0 & Hle0).
      rewrite Hu0 in Hp0, Hnd0, Hr0, Hle0. simpl in Hp0, Hnd0, Hr0, Hle0.
      destruct (IHrest (fun f u Hin k' n' => IH f u (or_intror Hin) k' n') n1)
        as ((rest' & Hshape) & Hp1 & Hnd1 & Hr1 & Hle1).
      destruct (rename_unboxed_field (Unboxed rest) n1) as [r n2] eqn:Hr.
      simpl in Hshape, Hp1, Hnd1, Hr1, Hle1.
      rewrite (Hshape rest eq_refl) in Hp1, Hnd1, Hr1 |- *. cbn [fst snd].
      split; [ exists ((f0, u0') :: rest'); reflexivity | ].
      split.
      { rewrite !ufield_paths_cons, Hp0. f_equal. exact Hp1. }
      rewrite ufield_leaves_cons, map_app.
      split; [ | split ].
      * apply NoDup_app; [ exact Hnd0 | exact Hnd1 | ].
        intros x Hx Hx'. destruct (Hr0 x Hx) as [_ Hlt].
        destruct (Hr1 x Hx') as [Hge _]. lia.
      * intros x Hx. apply in_app_or in Hx as [Hx | Hx];
          [ destruct (Hr0 x Hx) | destruct (Hr1 x Hx) ]; lia.
      * lia.
Qed.

(* A renaming keeps the fields: the field [f] of the renamed fields is a
   renaming of the field [f]. *)
Lemma rename_lookup (fields : unboxed_fields) :
  ∀ (n : positive) (f : field) (u : unboxed_field),
    Keyed.lookup field_eqb f fields = Some u ->
    ∃ (n' : positive) (fields' : unboxed_fields),
      fst (rename_unboxed_field (Unboxed fields) n) = Unboxed fields'
      ∧ Keyed.lookup field_eqb f fields'
        = Some (fst (rename_unboxed_field u n')).
Proof.
  induction fields as [ | [f0 u0] rest IH]; intros n f u Hl;
    [ discriminate Hl | ].
  rewrite rename_unboxed_cons.
  destruct (rename_unboxed_field u0 n) as [u0' n1] eqn:Hu0.
  destruct (rename_ok (Unboxed rest) Value n1) as ((rest' & Hshape) & _).
  pose proof (Hshape rest eq_refl) as Hrest.
  destruct (rename_unboxed_field (Unboxed rest) n1) as [r n2] eqn:Hr.
  cbn [fst] in Hrest. subst r.
  rewrite keyed_lookup_cons in Hl.
  destruct (field_eqb f f0) eqn:Hf.
  - injection Hl as <-. exists n, ((f0, u0') :: rest').
    split; [ reflexivity | ]. rewrite keyed_lookup_cons, Hf, Hu0.
    reflexivity.
  - destruct (IH n1 f u Hl) as (n' & fields' & Hfields' & Hl').
    rewrite Hr in Hfields'. cbn [fst] in Hfields'.
    injection Hfields' as <-.
    exists n', ((f0, u0') :: rest'). split; [ reflexivity | ].
    rewrite keyed_lookup_cons, Hf. exact Hl'.
Qed.

(* The pairs of [u0] with a renaming of [u] exist if those with [u] do. *)
Lemma pairs_rename (u0 : unboxed_field) :
  ∀ (k : flambda_kind) (u : unboxed_field) (n : positive)
    (pairs : list (variable * variable * flambda_kind)),
    unboxed_field_pairs k u0 u = Some pairs ->
    ∃ pairs' : list (variable * variable * flambda_kind),
      unboxed_field_pairs k u0 (fst (rename_unboxed_field u n)) = Some pairs'.
Proof.
  induction u0 as [x | fields0 IH] using unboxed_field_rec;
    intros k u n pairs Hp.
  - destruct u as [x' | fields]; [ | discriminate Hp ].
    simpl. eexists. reflexivity.
  - destruct u as [x' | fields]; [ discriminate Hp | ].
    destruct (rename_ok (Unboxed fields) Value n) as ((fields' & Hshape) & _).
    pose proof (Hshape fields eq_refl) as Hf'. rewrite Hf'.
    revert pairs Hp. induction fields0 as [ | [f u] rest IHrest];
      intros pairs Hp; [ eexists; reflexivity | ].
    change (unboxed_field_pairs k (Unboxed ((f, u) :: rest)) (Unboxed fields))
      with (match Keyed.lookup field_eqb f fields with
            | Some u' =>
              match unboxed_field_pairs (field_kind f) u u',
                    unboxed_field_pairs k (Unboxed rest) (Unboxed fields)
              with
              | Some p, Some q => Some (p ++ q)
              | _, _ => None
              end
            | None => None
            end) in Hp.
    change (unboxed_field_pairs k (Unboxed ((f, u) :: rest)) (Unboxed fields'))
      with (match Keyed.lookup field_eqb f fields' with
            | Some u' =>
              match unboxed_field_pairs (field_kind f) u u',
                    unboxed_field_pairs k (Unboxed rest) (Unboxed fields')
              with
              | Some p, Some q => Some (p ++ q)
              | _, _ => None
              end
            | None => None
            end).
    destruct (Keyed.lookup field_eqb f fields) as [uf | ] eqn:Hl;
      [ | discriminate Hp ].
    destruct (unboxed_field_pairs (field_kind f) u uf) as [p1 | ] eqn:Hp1;
      [ | discriminate Hp ].
    destruct (unboxed_field_pairs k (Unboxed rest) (Unboxed fields))
      as [q | ] eqn:Hq; [ | discriminate Hp ].
    destruct (rename_lookup fields n f uf Hl)
      as (n' & fields'' & Hf'' & Hl').
    rewrite Hf' in Hf''. injection Hf'' as <-. rewrite Hl'.
    destruct (IH f u (or_introl eq_refl) (field_kind f) uf n' p1 Hp1)
      as (p1' & Hp1').
    rewrite Hp1'.
    destruct (IHrest (fun f0 u0 Hin => IH f0 u0 (or_intror Hin)) q eq_refl)
      as (q' & Hq'). rewrite Hq'. eexists. reflexivity.
Qed.

(* The variables of fields with distinct variables, bound to the values
   of their leaves in a value, hold the value. *)
Lemma bind_fields_hold_gen (fields : unboxed_fields) (v : value)
    (ws : list value) (cv cv1 : env) :
  NoDup (leaf_variables fields) -> fields_vals fields v ws ->
  bind_params (leaf_variables fields) ws cv = Some cv1 ->
  fields_hold cv1 fields v.
Proof.
  intros Hnd Hws Hb path var k Hpath.
  destruct (upath_index _ _ _ _ _ Hpath) as (i & Hl & Hpi).
  destruct (forall2_nth_l _ _ _ i _ Hws Hpi) as (w' & Hw' & Hpw & Hk).
  exists w'. split; [ exact Hpw | split; [ | exact Hk ] ].
  apply (bind_params_nth _ ws cv cv1 i var w' Hnd Hb); [ | exact Hw' ].
  unfold leaf_variables. rewrite nth_error_map.
  change (unboxed_field_leaves Value (Unboxed fields))
    with (fields_leaves fields) in Hl.
  rewrite Hl. reflexivity.
Qed.

(* The variables that the pairs of [fields] with [fields'] read, in an
   environment where [fields'] hold a value, are the leaves of [fields]
   in it. *)
Lemma pairs_fields_vals (env0 : env) (fields fields' : unboxed_fields)
    (v : value) (pairs : list (variable * variable * flambda_kind)) :
  unboxed_field_pairs Value (Unboxed fields) (Unboxed fields') = Some pairs ->
  fields_hold env0 fields' v ->
  ∃ ws : list value,
    eval_simples env0 (map (fun '(_, x', _) => Var x') pairs) = Some ws
    ∧ fields_vals fields v ws.
Proof.
  intros Hp Hhold.
  exact (pairs_eval env0 fields' v pairs (fields_paths fields)
           (pairs_paths (Unboxed fields) Value (Unboxed fields') pairs Hp)
           Hhold).
Qed.

Section RebuildsCalls.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable pdc : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable ff ffu : positive.
Variable dead : list variable.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable kept_slots : value_slot -> bool.
Variable codes0 : list (code_id * code).
Variable symbols : list symbol.
Variable is_local : field -> bool.
Variable nr : code_id -> bool.
(* The code that the flow may bring to a variable. *)
Variable reach : bool -> variable -> code_id -> bool.
(* The code that a call may over-apply. *)
Variable ovr : code_id -> bool.
(* The code that what is left of an over-application may call. *)
Variable rem : code_id -> bool.
Hypothesis sol_closed : closed is_local (graph_of_facts F) sol.
Hypothesis local_slots : is_local_slots_only is_local.
Hypothesis fields_read : ∀ (x : variable) (fields : unboxed_fields)
    (f : field) (u : unboxed_field),
  fields_of unboxed x = Some fields -> (f, u) ∈ fields ->
  ∃ w base : node,
    g_accessor (graph_of_facts F) w f base
    ∧ s_usages sol (Node_variable x) base = true
    ∧ s_has_usage sol w = true.
Hypothesis reads_found : ∀ (x y : variable) (kind : flambda_kind)
    (f : field) (fields : unboxed_fields),
  fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
  g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
  is_block_like_field f = true ->
  is_dead_var sol x kind = false -> is_var_used sol x kind = true ->
  kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
  ∃ v : variable, Keyed.lookup field_eqb f fields = Some (Not_unboxed v).
Hypothesis dead_unused : ∀ z : variable,
  among z dead = true -> s_has_usage sol (Node_variable z) = false.

Abbreviation edges ctx ce e :=
  (FlambdaReaperFlowProofs.expr_edges (graph_of_facts F) noted deps cds
     kept_slots codes0 symbols True False ctx ce e).
Abbreviation sol_u := (solution_without_unboxed sol unboxed).
Abbreviation fresh z := (among z (field_variables unboxed) = false).

Definition cc_rebuild (en : rebuild_env) (e : expr) : expr :=
  rebuild_expr_cc F sol cannot_change code_deps pdc unit_code_ids codes ff en
    e.

Definition u_rebuild (en : unbox_env) (e : expr) : expr :=
  rebuild_expr_u true F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu [] en e.

(* The shape of a parameter that the first run keeps. *)
Definition pshape1 (p : pdesc) : pshape :=
  match pd_decision p with
  | Unbox fields => Psh_unboxed fields
  | Keep_unused kind => Psh_dead kind
  | Decided _ => Psh_plain
  end.

(* The shapes of the code of the unit. *)
Definition CSf (id : code_id) : option code_shape :=
  match lookup id codes, code_exn_decision sol code_deps unboxed id with
  | Some md, Some d =>
    match
      code_change_u sol cannot_change code_deps unit_code_ids unboxed id md,
      code_pdescs sol cannot_change code_deps unit_code_ids unboxed id md
    with
    | Changing_u pd rd, Some (pds, rds) =>
      Some
        (Code_shape (flat_map pshape_of pds) (flat_map pshape_of rds)
           (flat_map kept_kind rds)
           (arity_of_grouped_decisions_u
              (prepend_first [Decided Delete]
                 (group_by_parameter (cm_params_arity md) pd)))
           (arity_of_decisions_u rd) (pshape1 (Pdesc d true Value)) false)
    | Not_changing_u, Some (pds, rds) =>
      Some
        (Code_shape (flat_map pshape_of pds) (flat_map pshape_of rds)
           (flat_map kept_kind rds) (cm_params_arity md)
           (cm_result_arity md) (pshape1 (Pdesc d true Value)) true)
    | _, None => None
    end
  | _, _ => None
  end.

(* The code whose body is related with its return continuation out of the
   relation: the code that the fragment takes to never return. *)
Definition NRf (id : code_id) : bool := nr id.

(* The code that the flow may bring to a variable, as a relation. *)
Definition RCHf (b : bool) (f : variable) (id : code_id) : Prop :=
  reach b f id = true.

(* The code that a call may over-apply, as a relation. *)
Definition OVf (id : code_id) : Prop := ovr id = true.

(* The code that what is left of an over-application may call, as a
   relation. *)
Definition REMf (id : code_id) : Prop := rem id = true.

Record cinv (C : finite_map (list pdesc * bool)) (en : rebuild_env)
    (enu : unbox_env) : Prop := {
  ci_ok : ∀ (k : continuation) (ps : list pdesc) (h : bool),
    lookup k C = Some (ps, h) -> forallb pdesc_ok ps = true;
  ci_kinds : env_kinds (ue_env enu) = env_kinds en;
  ci_kinds_ok : ∀ x : variable, kind_ok (kind_of_variable en x) = true;
  ci_proj : env_projections en = [];
  ci_proj_u : env_projections (ue_env enu) = [];
  ci_drop : ue_drop_projections enu = false;
  ci_dec : ∀ k : continuation,
    lookup k (ue_decisions enu)
    = option_map (fun e => map pd_decision (fst e)) (lookup k C);
  (* The return continuation of code that keeps its calling convention, and
     the exception continuation of code, have no mask in the first run,
     which keeps all their parameters. *)
  ci_mask : ∀ k : continuation,
    lookup k (env_conts en)
    = option_map (fun e => map pd_keep (fst e)) (lookup k C)
    ∨ lookup k (env_conts en) = None
      ∧ ∃ (ps : list pdesc) (h : bool),
          lookup k C = Some (ps, h) ∧ forallb pd_keep ps = true
}.

(* The two terms related, the second with fewer free continuations (it
   drops the handler of a call that never returns) and binding fewer
   statics, its free continuations among [fc0] (the live ones of the
   original, [live_conts]). *)
Definition uc_result (K : finite_map kshape) (fc0 : list continuation)
    (p q : expr) : Prop :=
  uc unboxed dead CSf NRf RCHf OVf REMf K p q
  ∧ incl (free_continuations_of_expr q) (free_continuations_of_expr p)
  ∧ (binds_statics q = true -> binds_statics p = true)
  ∧ incl (free_continuations_of_expr q) fc0.

Lemma orb_impl2 (a a' b b' : bool) :
  (a' = true -> a = true) -> (b' = true -> b = true) ->
  a' || b' = true -> a || b = true.
Proof.
  intros Ha Hb H. apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (Ha H) | right; exact (Hb H) ].
Qed.

Lemma eq_incl_sym (l l' : list continuation) : l = l' -> incl l' l.
Proof. intros ->. intros x Hx. exact Hx. Qed.

Lemma incl_without_mono (b l l' : list continuation) :
  incl l l' -> incl (without b l) (without b l').
Proof.
  intros H k Hk. apply in_without_iff in Hk as [Hk Hn].
  apply in_without_iff. split; [ exact (H k Hk) | exact Hn ].
Qed.

Definition var_ok (z : variable) : Prop :=
  fresh z
  ∧ (s_has_usage sol (Node_variable z) = false ->
     fields_of unboxed z = None -> among z dead = true).

Lemma uc_result_invalid (K : finite_map kshape) (fc0 : list continuation) :
  uc_result K fc0 Invalid Invalid.
Proof.
  split; [ constructor | split; [ intros k [] | ] ].
  split; [ intros H; exact H | intros k [] ].
Qed.

(* A variable that the rebuild reads is used, so it is not one of those
   that the second run may give anything. *)
Lemma used_not_dead (z : variable) (kind : flambda_kind) :
  kind_ok kind = true -> is_var_used sol z kind = true ->
  among z dead = false.
Proof.
  intros Hk Hu. destruct (among z dead) eqn:Hd; [ | reflexivity ].
  unfold is_var_used in Hu. rewrite (dead_unused z Hd) in Hu.
  destruct kind; try discriminate Hk; discriminate Hu.
Qed.

Lemma rewrite_plain2 (en : rebuild_env) (s : simple) :
  (∀ x : variable, kind_ok (kind_of_variable en x) = true) ->
  plain_simple unboxed s = true ->
  plain_simple2 unboxed dead (rewrite_simple sol en s) = true.
Proof.
  intros Hk Hs. destruct s as [x | sym | c]; simpl in Hs |- *;
    [ | | reflexivity ].
  - destruct (is_var_used sol x (kind_of_variable en x)) eqn:Hu;
      [ | reflexivity ].
    simpl. unfold plain2. rewrite Hs.
    rewrite (used_not_dead x _ (Hk x) Hu). reflexivity.
  - destruct (is_symbol_used sol sym); [ exact Hs | reflexivity ].
Qed.

Lemma rewrite_plains2 (en : rebuild_env) (args : list simple) :
  (∀ x : variable, kind_ok (kind_of_variable en x) = true) ->
  forallb (plain_simple unboxed) args = true ->
  forallb (plain_simple2 unboxed dead) (map (rewrite_simple sol en) args)
  = true.
Proof.
  intros Hk. induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite !andb_true_iff. intros [Hs Hargs].
  split; [ apply rewrite_plain2; assumption | apply IH, Hargs ].
Qed.


Lemma kind_of_bind (en : rebuild_env) (bound : list (variable * flambda_kind))
    (x : variable) :
  (∀ y, kind_ok (kind_of_variable en y) = true) ->
  (∀ y kind, (y, kind) ∈ bound -> kind_ok kind = true) ->
  kind_ok (kind_of_variable (bind_variables en bound) x) = true.
Proof.
  intros Hen Hb. unfold kind_of_variable, bind_variables, bind_variables_then.
  cbn [env_kinds]. rewrite lookup_app.
  destruct (lookup x (rev bound)) as [kind | ] eqn:Hl.
  - apply (Hb x kind). apply in_rev.
    apply (keyed_lookup_in Pos.eqb Pos.eqb_eq) in Hl. exact Hl.
  - exact (Hen x).
Qed.

Lemma cinv_bind (C : finite_map (list pdesc * bool)) (en : rebuild_env)
    (enu : unbox_env) (bound : list (variable * flambda_kind)) :
  cinv C en enu ->
  (∀ y kind, (y, kind) ∈ bound -> kind_ok kind = true) ->
  cinv C (bind_variables en bound) (ubind_variables enu bound).
Proof.
  intros [Hok Hk Hkok Hp Hpu Hd Hdec Hmask] Hb. split.
  - exact Hok.
  - cbn [ue_env ubind_variables ubind_variables_then ubind_set
         bind_variables bind_variables_then env_kinds].
    rewrite Hk. reflexivity.
  - intros x. exact (kind_of_bind en bound x Hkok Hb).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - exact Hdec.
  - exact Hmask.
Qed.

Lemma lookup_map_ps {B : Type} (g : list pdesc -> B)
    (D : finite_map (list pdesc * bool)) (k : continuation) :
  lookup k (map (fun '(k, (ps, _)) => (k, g ps)) D)
  = option_map (fun e => g (fst e)) (lookup k D).
Proof.
  induction D as [ | [k' [ps h]] D IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma cinv_conts (C : finite_map (list pdesc * bool)) (en : rebuild_env)
    (enu : unbox_env) (D : finite_map (list pdesc * bool)) :
  cinv C en enu ->
  (∀ k ps h, (k, (ps, h)) ∈ D -> forallb pdesc_ok ps = true) ->
  cinv (D ++ C)
    (bind_continuations en (map (fun '(k, (ps, _)) => (k, map pd_keep ps)) D))
    (ubind_continuations enu
       (map (fun '(k, (ps, _)) => (k, map pd_decision ps)) D)).
Proof.
  intros [Hok Hk Hkok Hp Hpu Hd Hdec Hmask] HD. split.
  - intros k ps h Hl. rewrite lookup_app in Hl.
    destruct (lookup k D) as [[ps' h'] | ] eqn:HlD.
    + injection Hl as -> ->. apply (HD k ps h).
      apply (keyed_lookup_in Pos.eqb Pos.eqb_eq) in HlD. exact HlD.
    + exact (Hok k ps h Hl).
  - exact Hk.
  - exact Hkok.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros k. cbn [ue_decisions ubind_continuations].
    rewrite !lookup_app.
    rewrite (lookup_map_ps (map pd_decision) D k).
    destruct (lookup k D) as [[ps h] | ] eqn:HlD; [ reflexivity | ].
    apply Hdec.
  - intros k. cbn [env_conts bind_continuations].
    rewrite !lookup_app.
    rewrite (lookup_map_ps (map pd_keep) D k).
    destruct (lookup k D) as [[ps h] | ] eqn:HlD; [ left; reflexivity | ].
    destruct (Hmask k) as [Hm | (Hm & ps & h & HC & Hall)];
      [ left; exact Hm | right ].
    split; [ exact Hm | ]. exists ps, h. split; [ | exact Hall ].
    exact HC.
Qed.


Lemma strong_simple2 (s : simple) :
  FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed s = true ->
  (∀ z, z ∈ variables_in_simple s -> fresh z) ->
  plain_simple unboxed s = true.
Proof. apply strong_simple. Qed.

Lemma strong_simples2 (l : list simple) :
  forallb (FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed) l = true ->
  (∀ z, z ∈ variables_in_simples l -> fresh z) ->
  forallb (plain_simple unboxed) l = true.
Proof. apply strong_simples. Qed.

Lemma get_args_nil_dec (en : rebuild_env) (args : list simple) :
  get_args sol unboxed en [] args
  = Some (map (rewrite_simple sol_u en) args).
Proof.
  induction args as [ | a args IH]; [ reflexivity | ]. simpl. rewrite IH.
  reflexivity.
Qed.

Lemma map_rewrite_u (en : rebuild_env) (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  map (rewrite_simple sol_u en) args = map (rewrite_simple sol en) args.
Proof.
  intros H. apply map_ext_in. intros s Hs. apply rewrite_simple_u.
  rewrite forallb_forall in H. exact (H s Hs).
Qed.

Lemma filter_kept_nil_mask {A : Type} (l : list A) : filter_kept [] l = l.
Proof. induction l as [ | a l IH]; simpl; [ | rewrite IH ]; reflexivity. Qed.

Lemma filter_kept_all {A : Type} (l : list A) :
  ∀ mask : list bool,
    forallb (fun b => b) mask = true -> filter_kept mask l = l.
Proof.
  induction l as [ | a l IH]; intros [ | b mask] H; simpl;
    [ reflexivity | reflexivity | rewrite filter_kept_nil_mask; reflexivity
    | ].
  apply andb_true_iff in H as [-> H]. rewrite (IH mask H). reflexivity.
Qed.

Lemma forallb_pd_keep (ps : list pdesc) :
  forallb pd_keep ps = true -> forallb (fun b => b) (map pd_keep ps) = true.
Proof.
  induction ps as [ | p ps IH]; [ reflexivity | ]. cbn [map forallb].
  intros H. apply andb_true_iff in H as [-> H]. exact (IH H).
Qed.

(* An unboxed name is poison of its kind in the rebuild with unboxing. *)
Lemma rewrite_unboxed_poison (en : rebuild_env) (a : simple) :
  (∀ x : variable, kind_ok (kind_of_variable en x) = true) ->
  FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed a = false ->
  rewrite_simple sol_u en a = Const (Poison (simple_kind en a)).
Proof.
  intros Hk Hp.
  destruct a as [y | sym | c]; cbn in Hp; [ | | discriminate Hp ].
  - unfold FlambdaReaperUnboxingRefinesSpec.plain_variable in Hp.
    apply negb_false_iff in Hp.
    cbn [rewrite_simple simple_kind]. unfold is_var_used.
    specialize (Hk y).
    destruct (kind_of_variable en y); try (cbn in Hk; discriminate Hk);
      cbn [solution_without_unboxed s_has_usage is_unboxed_name];
      rewrite Hp; destruct (s_has_usage sol (Node_variable y));
      reflexivity.
  - apply negb_false_iff in Hp. cbn [rewrite_simple simple_kind].
    unfold is_symbol_used.
    cbn [solution_without_unboxed s_has_usage is_unboxed_name].
    rewrite Hp. destruct (s_has_usage sol (Node_symbol sym)); reflexivity.
Qed.

(* The arguments of an [Apply_cont] to a continuation described by
   [ps]. *)
Lemma get_args_rel2 (en : rebuild_env) (args : list simple) :
  (∀ x, kind_ok (kind_of_variable en x) = true) ->
  (∀ z, z ∈ variables_in_simples args -> fresh z) ->
  ∀ ps : list pdesc,
    forallb pdesc_ok ps = true ->
    args_ok2 sol unboxed (env_kinds en) ps args = true ->
    ∃ args_u : list simple,
      get_args sol unboxed en (map pd_decision ps) args = Some args_u
      ∧ args_rel2 unboxed dead (flat_map pshape_of ps)
          (filter_kept (map pd_keep ps) (map (rewrite_simple sol en) args))
          args_u.
Proof.
  intros Hkok. induction args as [ | a args IH]; intros Hfresh ps Hps Hok.
  - exists []. split; [ reflexivity | ]. cbn [map].
    replace (filter_kept (map pd_keep ps) []) with (@nil simple)
      by (destruct (map pd_keep ps); reflexivity).
    destruct (flat_map pshape_of ps) as [ | o sh ];
      [ apply args_rel2_beyond; reflexivity | apply args_rel2_missing ].
  - unfold variables_in_simples in Hfresh. cbn [flat_map] in Hfresh.
    assert (Ha : ∀ z, z ∈ variables_in_simple a -> fresh z)
      by (intros z Hz; apply Hfresh, in_or_app; left; exact Hz).
    assert (Hargs : ∀ z, z ∈ variables_in_simples args -> fresh z)
      by (intros z Hz; apply Hfresh, in_or_app; right; exact Hz).
    destruct ps as [ | p ps].
    + cbn [args_ok2] in Hok.
      pose proof (strong_simples2 (a :: args) Hok Hfresh) as Hstrong.
      exists (map (rewrite_simple sol_u en) (a :: args)).
      split; [ apply get_args_nil_dec | ].
      rewrite (map_rewrite_u en (a :: args) Hstrong).
      change (map pd_keep []) with (@nil bool).
      rewrite filter_kept_nil_mask. cbn [flat_map].
      apply args_rel2_beyond. apply rewrite_plains2; assumption.
    + cbn [args_ok2] in Hok. apply andb_true_iff in Hok as [Ha_ok Hok].
      cbn [forallb] in Hps. apply andb_true_iff in Hps as [Hp Hps].
      destruct (IH Hargs ps Hps Hok) as (rest & Hrest & Hrel).
      cbn [map get_args]. rewrite Hrest.
      unfold pdesc_ok in Hp. apply andb_true_iff in Hp as [_ Hp].
      destruct p as [d keep kind]. cbn [pd_decision pd_keep pd_kind] in *.
      unfold arg_ok in Ha_ok. cbn [pd_decision pd_keep] in Ha_ok.
      cbn [flat_map pshape_of pd_keep pd_decision].
      destruct d as [[kind' | ] | fields | kind'].
      * apply andb_true_iff in Hp as [Hkeep _]. subst keep.
        pose proof (strong_simple2 a Ha_ok Ha) as Hstrong.
        exists (rewrite_simple sol_u en a :: rest).
        split; [ reflexivity | ]. cbn [filter_kept app].
        rewrite (rewrite_simple_u sol unboxed en a Hstrong).
        apply args_rel2_plain; [ apply rewrite_plain2; assumption | ].
        exact Hrel.
      * apply negb_true_iff in Hp. subst keep.
        exists rest. split; [ reflexivity | ]. exact Hrel.
      * unfold unboxed_arg_ok in Ha_ok.
        destruct a as [y | sym | c]; try discriminate Ha_ok.
        destruct (fields_of unboxed y) as [fields_y | ] eqn:Hy;
          [ | discriminate Ha_ok ].
        apply andb_true_iff in Ha_ok as [Halike Hused].
        destruct (unboxed_args_pairs unboxed fields y fields_y Hy Halike)
          as (pairs & Hpairs & Hl).
        rewrite Hl. cbn [option_map].
        exists (map (fun '(_, x', _) => Var x') pairs ++ rest).
        split; [ rewrite map_map; do 2 f_equal; apply map_ext;
                 intros [[a b] k]; reflexivity | ].
        apply andb_true_iff in Hp as [_ Hp].
        destruct keep.
        -- cbn [app filter_kept]. simpl in Hused. unfold kinds_env in Hused.
           cbn [rewrite_simple]. unfold kind_of_variable in Hused |- *.
           cbn [env_kinds] in Hused. rewrite Hused.
           exact (args_rel2_unboxed unboxed dead _ fields fields_y y pairs _ _
                    Hy Hpairs Hrel).
        -- cbn [filter_kept app].
          rewrite (pairs_no_leaves fields fields_y pairs Hpairs
                     (no_leaves_nil fields Hp)).
          exact Hrel.
      * apply andb_true_iff in Hp as [Hkeep _]. subst keep.
        exists (rewrite_simple sol_u en a :: rest).
        split; [ reflexivity | ]. cbn [filter_kept app].
        apply args_rel2_dead; [ | exact Hrel ].
        destruct (FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed a)
          eqn:Hpl.
        -- right. pose proof (strong_simple2 a Hpl Ha) as Hstrong.
           rewrite (rewrite_simple_u sol unboxed en a Hstrong).
           split; [ reflexivity | apply rewrite_plain2; assumption ].
        -- left. rewrite (rewrite_unboxed_poison en a Hkok Hpl).
           cbn [orb] in Ha_ok. apply kind_eqb_eq in Ha_ok.
           rewrite <- Ha_ok. destruct a; reflexivity.
Qed.

Lemma rewrite_apply_cont_rel2 (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (ac : apply_cont) :
  cinv C en enu ->
  (∀ z, z ∈ variables_in_apply_cont ac -> fresh z) ->
  apply_cont_ok2 sol unboxed (env_kinds en) C ac = true ->
  (rewrite_apply_cont sol en ac = None
   ∧ rewrite_apply_cont_u sol unboxed enu ac = None)
  ∨ ∃ ac_p ac_u : apply_cont,
      rewrite_apply_cont sol en ac = Some ac_p
      ∧ rewrite_apply_cont_u sol unboxed enu ac = Some ac_u
      ∧ ac_rel2 unboxed dead (K_of C) ac_p ac_u.
Proof.
  intros Hci Hfresh Hok. unfold rewrite_apply_cont, rewrite_apply_cont_u.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ left; split; reflexivity | right ].
  unfold apply_cont_ok2 in Hok.
  assert (Hmap : map (rewrite_simple sol (ue_env enu)) (ac_args ac)
                 = map (rewrite_simple sol en) (ac_args ac))
    by (apply map_ext; intros s;
        apply rewrite_simple_kinds, (ci_kinds _ _ _ Hci)).
  pose proof (ci_mask _ _ _ Hci (ac_continuation ac)) as Hm.
  rewrite (ci_dec _ _ _ Hci (ac_continuation ac)).
  unfold ac_rel2, kshape_at.
  destruct (lookup (ac_continuation ac) C) as [[ps hb] | ] eqn:HCk.
  - assert (Hfk : ∀ l : list simple,
              match lookup (ac_continuation ac) (env_conts en) with
              | Some mask => filter_kept mask l
              | None => l
              end
              = filter_kept (map pd_keep ps) l).
    { intros l. destruct Hm as [-> | (-> & ps' & h' & HC' & Hall)];
        [ reflexivity | ].
      injection HC' as <- _. symmetry.
      exact (filter_kept_all l _ (forallb_pd_keep ps Hall)). }
    rewrite Hfk. cbn [option_map fst].
    destruct (get_args_rel2 (ue_env enu) (ac_args ac)) with (ps := ps)
      as (args_u & Hget & Hrel).
    + intros x. unfold kind_of_variable. rewrite (ci_kinds _ _ _ Hci).
      exact (ci_kinds_ok _ _ _ Hci x).
    + exact Hfresh.
    + exact (ci_ok _ _ _ Hci _ _ _ HCk).
    + rewrite (ci_kinds _ _ _ Hci). exact Hok.
    + rewrite Hget. eexists _, _.
      split; [ reflexivity | split; [ reflexivity | ] ].
      cbn [ac_continuation ac_args ac_trap_action].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ | apply K_of_not_never ].
      rewrite lookup_K_of, HCk. cbn [option_map fst ks_shapes].
      rewrite <- Hmap. exact Hrel.
  - destruct Hm as [Hm | (_ & ps' & h' & HC' & _)];
      [ rewrite Hm | discriminate HC' ].
    cbn [option_map].
    pose proof (strong_simples2 _ Hok Hfresh) as Hstrong.
    rewrite (map_rewrite_u (ue_env enu) _ Hstrong), Hmap.
    eexists _, _. split; [ reflexivity | split; [ reflexivity | ] ].
    cbn [ac_continuation ac_args ac_trap_action].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ | apply K_of_not_never ].
    rewrite lookup_K_of, HCk. cbn [option_map].
    apply args_rel2_beyond, rewrite_plains2;
      [ exact (ci_kinds_ok _ _ _ Hci) | exact Hstrong ].
Qed.


(* ------------------------------------------------------------------------ *)
(* Exceptions                                                               *)
(* ------------------------------------------------------------------------ *)

(* The decision of the exception continuation of code: kept, with a use
   or not, as a continuation of fixed arity. *)
Lemma code_exn_decision_cases (id : code_id) (d : decision) :
  code_exn_decision sol code_deps unboxed id = Some d ->
  (d = Decided (Keep Value) ∨ d = Keep_unused Value)
  ∧ ∃ cd : code_dep,
      lookup id code_deps = Some cd
      ∧ fixed_arity_decisions sol unboxed [cd_exn cd] [Value] = [d].
Proof.
  unfold code_exn_decision. intros H.
  destruct (lookup id code_deps) as [cd | ]; [ | discriminate H ].
  split; [ | exists cd; split; [ reflexivity | ] ];
    revert H; unfold fixed_arity_decisions; cbn [combine map];
    (destruct (unboxed_of_node unboxed (cd_exn cd)) as [fields | ];
     [ intros H; discriminate H | ]);
    destruct (is_node_used sol (cd_exn cd) Value); intros H;
    injection H as <-; first [ left; reflexivity | right; reflexivity
                             | reflexivity ].
Qed.

(* The parameter that takes the exception of a callee is kept by the
   first run, a value, and of a shape that the exception fits. *)
Lemma exn_param_fits (k : pdesc) (d : decision) :
  (d = Decided (Keep Value) ∨ d = Keep_unused Value) ->
  pdesc_ok k = true -> exn_param_ok k (Pdesc d true Value) = true ->
  pd_keep k = true ∧ pd_kind k = Value
  ∧ fits (pshape1 (Pdesc d true Value)) (pshape1 k).
Proof.
  intros Hd Hok Hpar. destruct k as [dk keep kind].
  unfold exn_param_ok in Hpar. cbn [pd_keep pd_decision] in Hpar |- *.
  apply andb_true_iff in Hpar as [Hpar Hres].
  apply andb_true_iff in Hpar as [Hkeep Hnu]. subst keep.
  unfold result_ok in Hres. cbn [pd_kind pd_decision] in Hres.
  apply andb_true_iff in Hres as [Hkind Hres].
  apply kind_eqb_eq in Hkind. subst kind.
  split; [ reflexivity | split; [ reflexivity | ] ].
  unfold pdesc_ok in Hok. cbn [pd_kind pd_decision pd_keep] in Hok.
  unfold pshape1. cbn [pd_decision].
  destruct dk as [[kk | ] | fields | kk]; try discriminate Hnu;
    try discriminate Hok;
    destruct Hd as [-> | ->]; try discriminate Hres; try constructor.
  apply andb_true_iff in Hok as [_ Hok]. apply andb_true_iff in Hok as [_ Hk].
  apply kind_eqb_eq in Hk. subst kk. constructor.
Qed.

(* The extra arguments that the rebuild with unboxing gives are the
   arguments that it gives an [Apply_cont]. *)
Lemma get_extra_args_fst (en : rebuild_env)
    (extra : list (simple * flambda_kind)) :
  ∀ ds : list decision,
    option_map (map fst) (get_extra_args sol unboxed en ds extra)
    = get_args sol unboxed en ds (map fst extra).
Proof.
  induction extra as [ | [s kind] extra IH]; intros ds; [ reflexivity | ].
  destruct ds as [ | d ds].
  - cbn [get_extra_args get_args map fst]. rewrite <- (IH []).
    destruct (get_extra_args sol unboxed en [] extra); reflexivity.
  - cbn [get_extra_args get_args map fst]. rewrite <- (IH ds).
    destruct (get_extra_args sol unboxed en ds extra) as [rest | ];
      [ | reflexivity ].
    cbn [option_map].
    destruct d as [[k | ] | fields | k]; try reflexivity.
    destruct (unboxed_args unboxed fields s); [ | reflexivity ].
    cbn [option_map]. rewrite map_app. reflexivity.
Qed.

Lemma map_fst_filter_kept {A B : Type} (l : list (A * B)) :
  ∀ mask : list bool,
    map fst (filter_kept mask l) = filter_kept mask (map fst l).
Proof.
  induction l as [ | a l IH]; intros [ | b mask]; cbn [filter_kept map];
    [ reflexivity | reflexivity | rewrite IH; reflexivity | ].
  destruct b; cbn [map]; rewrite IH; reflexivity.
Qed.

Lemma filter_kept_kinds_length {A : Type} (ps : list pdesc) :
  ∀ l : list A,
    length l = length ps ->
    length (filter_kept (map pd_keep ps) l) = length (flat_map kept_kind ps).
Proof.
  induction ps as [ | p ps IH]; intros [ | a l] Hl; cbn in Hl;
    try discriminate Hl; [ reflexivity | ].
  cbn [map filter_kept flat_map]. unfold kept_kind at 1.
  destruct (pd_keep p); cbn [length app]; rewrite IH by lia; reflexivity.
Qed.

(* The exception continuation of a call, with its extra arguments, as the
   two rebuilds make them. *)
Lemma exn_rebuilt (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (d : decision) :
  cinv C en enu ->
  (d = Decided (Keep Value) ∨ d = Keep_unused Value) ->
  exn_cont_ok sol unboxed (env_kinds en) C (ap_exn_continuation ap)
    (Pdesc d true Value) (ap_exn_extra_args ap)
  = true ->
  (∀ z, z ∈ variables_in_simples (exn_extra_simples ap) -> fresh z) ->
  ∃ extra_u : list (simple * flambda_kind),
    rebuild_exn_extra_args_u sol unboxed enu ap = Some extra_u
    ∧ ∀ ap_cc ap_u : apply_expr,
        ap_exn_continuation ap_cc = ap_exn_continuation ap ->
        ap_exn_extra_args ap_cc = rebuild_exn_extra_args sol en ap ->
        ap_exn_extra_args ap_u = extra_u ->
        exn_call_rel unboxed dead (K_of C) (pshape1 (Pdesc d true Value))
          ap_cc ap_u.
Proof.
  intros Hci Hd Hok Hfresh.
  unfold exn_cont_ok in Hok.
  unfold rebuild_exn_extra_args_u, rebuild_exn_extra_args.
  rewrite (ci_dec _ _ _ Hci (ap_exn_continuation ap)).
  pose proof (ci_mask _ _ _ Hci (ap_exn_continuation ap)) as Hm.
  destruct (lookup (ap_exn_continuation ap) C) as [[ps hb] | ] eqn:HC.
  - destruct ps as [ | k ps]; [ destruct hb; discriminate Hok | ].
    pose proof (ci_ok _ _ _ Hci _ _ _ HC) as Hps_ok.
    cbn [forallb] in Hps_ok. apply andb_true_iff in Hps_ok as [Hk_ok Hps_ok].
    destruct hb.
    + (* A handler. *)
      assert (Hok' : exn_param_ok k (Pdesc d true Value)
                     && Nat.eqb (length (ap_exn_extra_args ap)) (length ps)
                     && args_ok2 sol unboxed (env_kinds en) ps
                          (map fst (ap_exn_extra_args ap))
                     = true)
        by (destruct ps; exact Hok).
      clear Hok. rename Hok' into Hok.
      apply andb_true_iff in Hok as [Hok Hargs].
      apply andb_true_iff in Hok as [Hpar Hlen]. apply Nat.eqb_eq in Hlen.
      destruct (exn_param_fits k d Hd Hk_ok Hpar) as (Hkeep & Hkind & Hfits).
      cbn [option_map fst map tl].
      destruct (get_args_rel2 (ue_env enu) (exn_extra_simples ap)) with
          (ps := ps)
        as (args_u & Hget & Hrel).
      { intros x. unfold kind_of_variable. rewrite (ci_kinds _ _ _ Hci).
        exact (ci_kinds_ok _ _ _ Hci x). }
      { exact Hfresh. }
      { exact Hps_ok. }
      { rewrite (ci_kinds _ _ _ Hci). exact Hargs. }
      unfold exn_extra_simples in Hget.
      rewrite <- get_extra_args_fst in Hget.
      destruct (get_extra_args sol unboxed (ue_env enu) (map pd_decision ps)
                  (ap_exn_extra_args ap))
        as [extra_u | ] eqn:He; cbn [option_map] in Hget;
        [ | discriminate Hget ].
      injection Hget as Hget.
      exists extra_u. split; [ reflexivity | ].
      intros ap_cc ap_u Hecc Hxcc Hxu. right.
      exists (pshape1 k), (flat_map pshape_of ps), (flat_map kept_kind ps).
      assert (Hx : ap_exn_extra_args ap_cc
                   = filter_kept (map pd_keep ps)
                       (map
                          (fun '(s, kind) => (rewrite_simple sol en s, kind))
                          (ap_exn_extra_args ap))).
      { rewrite Hxcc.
        destruct Hm as [-> | (-> & ps' & h' & HC' & Hall)]; [ reflexivity | ].
        injection HC' as <- _. cbn [forallb] in Hall.
        apply andb_true_iff in Hall as [_ Hall]. symmetry.
        exact (filter_kept_all _ _ (forallb_pd_keep ps Hall)). }
      rewrite Hecc, lookup_K_of, HC.
      split.
      { destruct k as [dk keep kind]. cbn in Hkeep, Hkind. subst keep kind.
        reflexivity. }
      split; [ exact Hfits | ].
      split.
      * unfold exn_extra_simples. rewrite Hx, Hxu, Hget, map_fst_filter_kept.
        assert (Hmap : map fst
                         (map
                            (fun '(s, kind) => (rewrite_simple sol en s, kind))
                            (ap_exn_extra_args ap))
                       = map (rewrite_simple sol (ue_env enu))
                           (map fst (ap_exn_extra_args ap))).
        { rewrite !map_map. apply map_ext. intros [s kind]. cbn [fst].
          symmetry. apply rewrite_simple_kinds, (ci_kinds _ _ _ Hci). }
        rewrite Hmap. exact Hrel.
      * rewrite Hx. apply filter_kept_kinds_length.
        rewrite length_map. exact Hlen.
    + (* The exception continuation of the code the call is in. *)
      destruct ps as [ | ? ? ]; [ | discriminate Hok ].
      apply andb_true_iff in Hok as [Hpar Hnil].
      destruct (ap_exn_extra_args ap) as [ | ? ? ] eqn:Hx;
        [ | discriminate Hnil ].
      destruct (exn_param_fits k d Hd Hk_ok Hpar) as (Hkeep & Hkind & Hfits).
      exists []. split; [ reflexivity | ].
      intros ap_cc ap_u Hecc Hxcc Hxu. left.
      split; [ rewrite Hxcc;
               destruct (lookup (ap_exn_continuation ap) (env_conts en));
               reflexivity | ].
      split; [ exact Hxu | ].
      right. exists (pshape1 k).
      rewrite Hecc, lookup_K_of, HC.
      split; [ | exact Hfits ].
      destruct k as [dk keep kind]. cbn in Hkeep, Hkind. subst keep kind.
      reflexivity.
  - (* A way out. *)
    apply andb_true_iff in Hok as [Hpar Hnil].
    cbn [exn_pdescs forallb2] in Hpar. rewrite andb_true_r in Hpar.
    destruct (exn_param_fits (Pdesc (Decided (Keep Value)) true Value) d Hd
                eq_refl Hpar)
      as (_ & _ & Hfits).
    destruct (ap_exn_extra_args ap) as [ | ? ? ] eqn:Hx;
      [ | discriminate Hnil ].
    exists []. split; [ reflexivity | ].
    intros ap_cc ap_u Hecc Hxcc Hxu. left.
    split; [ rewrite Hxcc;
             destruct (lookup (ap_exn_continuation ap) (env_conts en));
             reflexivity | ].
    split; [ exact Hxu | ].
    left. rewrite Hecc, lookup_K_of, HC.
    split; [ reflexivity | ].
    destruct Hd as [-> | ->]; [ reflexivity | ].
    cbn in Hfits. inversion Hfits.
Qed.

Lemma plain2_of (x : variable) (kind : flambda_kind) :
  fields_of unboxed x = None -> fresh x -> kind_ok kind = true ->
  is_var_used sol x kind = true -> plain2 unboxed dead x = true.
Proof.
  intros Hx Hf Hk Hu. unfold plain2.
  rewrite (used_not_dead x kind Hk Hu), andb_true_r.
  apply plain_variable_iff. split; [ | exact Hf ].
  unfold Keyed.is_bound. unfold fields_of in Hx. rewrite Hx. reflexivity.
Qed.

Definition decide_param (p : (variable * flambda_kind) * bool) : decision :=
  let '((x, kind), keep) := p in
  match unboxed_of_node unboxed (Node_variable x) with
  | Some fields => Unbox fields
  | None =>
    if true && keep && negb (is_var_used sol x kind)
    then Keep_unused kind
    else Decided (if keep then Keep kind else Delete)
  end.

(* The parameters of a handler, as the two rebuilds make them. *)
Lemma params_rel2_gen (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    (∀ x kind, (x, kind) ∈ bound -> var_ok x ∧ kind_ok kind = true) ->
    let ds := map decide_param (combine bound mask) in
    let ps :=
      map (fun '(d, keep, kind) => Pdesc d keep kind)
        (combine (combine ds mask) (map snd bound))
    in
    forallb pdesc_ok ps = true ->
    params_rel2 unboxed dead (flat_map pshape_of ps) (filter_kept mask bound)
      (params_of_decisions bound ds)
    ∧ map pd_decision ps = ds ∧ map pd_keep ps = mask
    ∧ map snd (filter_kept mask bound) = flat_map kept_kind ps.
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] Hlen Hb;
    simpl in Hlen; try discriminate Hlen; cbv zeta; intros Hok.
  - split; [ constructor | split; [ reflexivity | split; reflexivity ] ].
  - cbn [combine map forallb] in Hok. apply andb_true_iff in Hok as [Hp Hok].
    destruct (Hb x kind (or_introl eq_refl)) as [[Hfresh Hdead] Hkind].
    destruct (IH mask ltac:(lia)
                (fun y k Hin => Hb y k (or_intror Hin)) Hok)
      as (Hrel & Hds & Hkeeps & Hkinds).
    cbn [combine map flat_map pd_decision pd_keep].
    unfold pdesc_ok in Hp. cbn [pd_kind pd_decision pd_keep] in Hp.
    cbn [decide_param] in Hp |- *.
    change (unboxed_of_node unboxed (Node_variable x))
      with (fields_of unboxed x) in Hp |- *.
    destruct (fields_of unboxed x) as [fields | ] eqn:Hx.
    + apply andb_true_iff in Hp as [_ Hp].
      apply andb_true_iff in Hp as [Hv Hkeep]. apply kind_eqb_eq in Hv.
      cbn [snd] in Hv. subst kind.
      split; [ | split; [ f_equal; exact Hds
                        | split; [ f_equal; exact Hkeeps | ] ] ].
      * destruct keep.
        -- cbn [filter_kept params_of_decisions decision_params pshape_of
                pd_keep pd_decision app].
           apply params_rel2_unboxed; assumption.
        -- cbn [filter_kept params_of_decisions decision_params pshape_of
                pd_keep app].
           rewrite (no_leaves_nil fields Hkeep). exact Hrel.
      * destruct keep; cbn [filter_kept kept_kind pd_keep pd_kind map app];
          [ f_equal | ]; exact Hkinds.
    + destruct keep eqn:Hk; cbn [andb negb].
      * destruct (is_var_used sol x kind) eqn:Hused; cbn [negb].
        -- split; [ | split; [ f_equal; exact Hds
                             | split; [ f_equal; exact Hkeeps | ] ] ].
           ++ cbn [filter_kept params_of_decisions decision_params pshape_of
                   pd_keep pd_decision app].
              apply params_rel2_plain; [ | exact Hrel ].
              exact (plain2_of x kind Hx Hfresh Hkind Hused).
           ++ cbn [filter_kept kept_kind pd_keep pd_kind map app].
              f_equal. exact Hkinds.
        -- split; [ | split; [ f_equal; exact Hds
                             | split; [ f_equal; exact Hkeeps | ] ] ].
           ++ cbn [filter_kept params_of_decisions decision_params pshape_of
                   pd_keep pd_decision app].
              apply params_rel2_dead;
                [ | exact Hx | exact Hfresh | exact Hrel ].
              apply Hdead; [ | reflexivity ].
              unfold is_var_used in Hused.
              destruct kind; try discriminate Hkind; exact Hused.
           ++ cbn [filter_kept kept_kind pd_keep pd_kind map app].
              f_equal. exact Hkinds.
      * split; [ | split; [ f_equal; exact Hds
                          | split; [ f_equal; exact Hkeeps | exact Hkinds ] ] ].
        cbn [filter_kept params_of_decisions decision_params pshape_of
             pd_keep app].
        exact Hrel.
Qed.


Lemma rewrite_var_used (en : rebuild_env) (s : simple) (z : variable) :
  rewrite_simple sol en s = Var z ->
  is_var_used sol z (kind_of_variable en z) = true.
Proof.
  destruct s as [x | sym | c]; simpl; intros H.
  - destruct (is_var_used sol x (kind_of_variable en x)) eqn:Hu;
      [ injection H as <-; exact Hu | discriminate H ].
  - destruct (is_symbol_used sol sym); discriminate H.
  - discriminate H.
Qed.

Lemma simple2_of_used (en : rebuild_env) (s : simple) :
  (∀ x, kind_ok (kind_of_variable en x) = true) ->
  plain_simple unboxed s = true ->
  (∀ z, s = Var z -> is_var_used sol z (kind_of_variable en z) = true) ->
  plain_simple2 unboxed dead s = true.
Proof.
  intros Hk Hs Hu. destruct s as [x | sym | c]; simpl in Hs |- *;
    [ | exact Hs | reflexivity ].
  unfold plain2. rewrite Hs.
  rewrite (used_not_dead x _ (Hk x) (Hu x eq_refl)). reflexivity.
Qed.

Lemma named_plain2_rebuilt (en : rebuild_env) (x : variable) (d : named) :
  (∀ y, kind_ok (kind_of_variable en y) = true) ->
  named_plain unboxed d = true ->
  named_plain2 unboxed dead (rebuild_named sol en x d) = true.
Proof.
  intros Hk Hd.
  assert (Hm : ∀ args, forallb (plain_simple unboxed) args = true ->
            forallb (plain_simple2 unboxed dead)
              (map (rewrite_simple sol en) args) = true)
    by (intros args Ha; apply rewrite_plains2; assumption).
  assert (Hf : ∀ args, forallb (plain_simple unboxed) args = true ->
            ∀ i, forallb (plain_simple2 unboxed dead)
                   (rebuild_fields sol en (Node_variable x) i args) = true).
  { intros args. induction args as [ | a args IH]; intros Ha i;
      [ reflexivity | ].
    simpl in Ha |- *. apply andb_true_iff in Ha as [Ha Hargs].
    rewrite (IH Hargs), andb_true_r.
    destruct (field_used sol _ _); [ | reflexivity ].
    apply rewrite_plain2; assumption. }
  destruct d as [s | p args | | | | | ]; simpl in Hd; try discriminate Hd.
  - simpl. apply rewrite_plain2; assumption.
  - destruct p as [p | p | p | p | p]; simpl; try (apply Hm, Hd).
    + destruct p; simpl; try (apply Hm, Hd).
      destruct (field_used sol _ _); simpl; [ apply Hm, Hd | reflexivity ].
    + destruct p; simpl; try (apply Hm, Hd). apply Hf, Hd.
    + destruct p; simpl; try (apply Hm, Hd). apply Hf, Hd.
Qed.

(* The fields of an allocation that the rebuild keeps are of variables
   that are used. *)
Lemma alloc_entries_to2 (en : rebuild_env) (x : variable) (d : named)
    (fields : unboxed_fields)
    (ls : list (list (variable * flambda_kind * simple))) :
  (∀ y, kind_ok (kind_of_variable en y) = true) ->
  (∀ (f : field) (kind : flambda_kind) (s : simple) (z : variable),
     alloc_field d f kind s -> s = Var z ->
     is_var_used sol z (kind_of_variable en z) = true) ->
  Forall2 (alloc_entry unboxed x d) fields ls ->
  Forall2 (alloc_entry2 unboxed dead x d) fields ls.
Proof.
  intros Hk Hu. induction 1 as [ | [f u] l fields ls Hfu _ IH];
    constructor; [ | exact IH ].
  destruct u as [v | nested]; cbn [alloc_entry alloc_entry2] in Hfu |- *.
  - destruct Hfu as (kind & s & -> & Hf & Hs). exists kind, s.
    split; [ reflexivity | split; [ exact Hf | ] ].
    apply (simple2_of_used en); [ exact Hk | exact Hs | ].
    intros z ->. exact (Hu f kind (Var z) z Hf eq_refl).
  - exact Hfu.
Qed.

Lemma rebuild_fields_used (en : rebuild_env) (base : node)
    (args : list simple) :
  ∀ (i j : nat) (s : simple) (z : variable),
    nth_error (rebuild_fields sol en base i args) j = Some s -> s = Var z ->
    is_var_used sol z (kind_of_variable en z) = true.
Proof.
  induction args as [ | a args IH]; intros i j s z Hn Hs;
    [ destruct j; discriminate Hn | ].
  destruct j as [ | j]; simpl in Hn.
  - injection Hn as Hn. rewrite Hs in Hn.
    destruct (field_used sol base _); [ | discriminate Hn ].
    exact (rewrite_var_used en a z Hn).
  - exact (IH (S i) j s z Hn Hs).
Qed.

Lemma read_of_block_like (p : unary_primitive) (y : variable) (f : field) :
  read_of (Prim (Unary p) [Var y]) = Some (f, Var y) ->
  is_block_like_field f = true.
Proof.
  intros H. destruct p; try discriminate H; simpl in H;
    injection H as <-; reflexivity.
Qed.


Lemma cc_rebuild_let_static (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  cc_rebuild en (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding_cc F sol cannot_change code_deps pdc unit_code_ids
            codes ff en)
         bindings)
      (cc_rebuild (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma u_rebuild_let_static (enu : unbox_env)
    (bindings : list static_binding) (body : expr) :
  u_rebuild enu (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding_u true F sol cannot_change code_deps pdc
            unit_code_ids codes unboxed ffu [] enu)
         bindings)
      (u_rebuild (ubind_variables enu []) body).
Proof. reflexivity. Qed.

Lemma cc_rebuild_let_cont (en : rebuild_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  cc_rebuild en (Let_cont_non_recursive k h body)
  = let mask := handler_mask sol (handler_is_exn_handler h) h in
    let body := cc_rebuild (bind_continuations en [(k, mask)]) body in
    if among k (free_continuations_of_expr body)
    then
      Let_cont_non_recursive k
        (rebuild_handler_cc F sol cannot_change code_deps pdc unit_code_ids
           codes ff en true mask h)
        body
    else body.
Proof. reflexivity. Qed.

Lemma u_rebuild_let_cont (enu : unbox_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  u_rebuild enu (Let_cont_non_recursive k h body)
  = let decisions :=
      handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    let body :=
      u_rebuild (ubind_continuations enu [(k, decisions)]) body in
    if among k (free_continuations_of_expr body)
    then
      Let_cont_non_recursive k
        (rebuild_handler_u true F sol cannot_change code_deps pdc
           unit_code_ids codes unboxed ffu [] enu true decisions h)
        body
    else body.
Proof. reflexivity. Qed.

Lemma cc_rebuild_let_cont_rec (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  cc_rebuild en (Let_cont_recursive handlers body)
  = let en :=
      bind_continuations en
        (map (fun '(k, h) => (k, handler_mask sol false h)) handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             rebuild_handler_cc F sol cannot_change code_deps pdc
               unit_code_ids codes ff en false (handler_mask sol false h) h))
         handlers)
      (cc_rebuild en body).
Proof. reflexivity. Qed.

Lemma u_rebuild_let_cont_rec (enu : unbox_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  u_rebuild enu (Let_cont_recursive handlers body)
  = let enu :=
      ubind_continuations enu
        (map (fun '(k, h) => (k, handler_decisions true sol unboxed false h))
           handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             rebuild_handler_u true F sol cannot_change code_deps pdc
               unit_code_ids codes unboxed ffu [] enu false
               (handler_decisions true sol unboxed false h) h))
         handlers)
      (u_rebuild enu body).
Proof. reflexivity. Qed.

Lemma cc_rebuild_handler (en : rebuild_env) (dead_check : bool)
    (mask : list bool) (params : list variable) (metadata : handler_metadata)
    (handler : expr) :
  rebuild_handler_cc F sol cannot_change code_deps pdc unit_code_ids codes ff
    en dead_check mask (Handler params metadata handler)
  = let bound := with_kinds params (hm_params_kinds metadata) in
    Handler (filter_kept mask params)
      (with_params_kinds metadata (filter_kept mask (hm_params_kinds metadata)))
      (if dead_check && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else cc_rebuild (bind_variables en bound) handler).
Proof. reflexivity. Qed.

Lemma u_rebuild_handler (enu : unbox_env) (dead_check : bool)
    (decisions : list decision) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  rebuild_handler_u true F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu [] enu dead_check decisions
    (Handler params metadata handler)
  = let bound := with_kinds params (hm_params_kinds metadata) in
    let params := params_of_decisions bound decisions in
    Handler (map fst params) (with_params_kinds metadata (map snd params))
      (if dead_check && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else u_rebuild (ubind_variables enu bound) handler).
Proof. reflexivity. Qed.

Definition names_ok (e : expr) : Prop :=
  (∀ z : variable, z ∈ variables_of e -> var_ok z)
  ∧ (∀ k : continuation, k ∈ continuations_of e -> k <> ff ∧ k <> ffu).

Definition names_ok_h (h : continuation_handler) : Prop :=
  (∀ z : variable, z ∈ variables_in_handler h -> var_ok z)
  ∧ (∀ k : continuation,
       k ∈ continuations_in_handler h -> k <> ff ∧ k <> ffu).

Definition rebuild_ok_expr (e : expr) : Prop :=
  ∀ (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (ctx : option code_id)
    (ce : finite_map (list node)),
    cinv C en enu ->
    calls_expr_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
      nr reach ovr rem
      (env_kinds en) C e
    = true ->
    names_ok e -> edges ctx ce e ->
    uc_result (K_of C) (live_conts F sol code_deps codes e) (cc_rebuild en e)
      (u_rebuild enu e).

Definition code_link (d : decision) (dcc : param_decision) : Prop :=
  match d with
  | Decided dd => dcc = dd
  | Unbox _ => True
  | Keep_unused _ => False
  end.

Lemma decisions_link (nodes : list node) (kinds : list flambda_kind) :
  Forall2 code_link (decisions_u sol unboxed nodes kinds)
    (decisions_of sol nodes kinds).
Proof.
  unfold decisions_u, decisions_of.
  induction (combine nodes kinds) as [ | [n k] l IH]; [ constructor | ].
  cbn [map]. constructor; [ | exact IH ].
  destruct (unboxed_of_node unboxed n); [ exact I | reflexivity ].
Qed.

Lemma pdescs_of_length (ds : list decision) (keeps : list bool)
    (kinds : list flambda_kind) :
  length (pdescs_of ds keeps kinds)
  = Nat.min (length ds) (Nat.min (length keeps) (length kinds)).
Proof.
  revert keeps kinds. induction ds as [ | d ds IH]; intros [ | b keeps]
    [ | k kinds]; simpl; try reflexivity.
  rewrite IH. reflexivity.
Qed.

Lemma same_paths_eq (fields fields' : unboxed_fields) :
  same_paths fields fields' = true ->
  fields_paths fields = fields_paths fields'.
Proof. apply paths_eqb_eq. Qed.

Lemma leaves_of_paths_nil (fields : unboxed_fields) :
  fields_paths fields = [] -> fields_leaves fields = [].
Proof.
  intros H. pose proof (Forall2_length (fields_leaves_paths fields)) as Hl.
  rewrite H in Hl. destruct (fields_leaves fields); [ reflexivity | ].
  discriminate Hl.
Qed.

(* The parameters of code, as the two rebuilds make them. *)
Lemma code_params_rel2 (params : list variable) :
  ∀ (pd : list decision) (pd_cc : list param_decision)
    (kinds : list flambda_kind),
    Forall2 code_link pd pd_cc ->
    length params = length (pdescs_of pd (map is_kept pd_cc) kinds) ->
    forallb pdesc_ok (pdescs_of pd (map is_kept pd_cc) kinds) = true ->
    forallb2 (code_param_ok unboxed) params
      (pdescs_of pd (map is_kept pd_cc) kinds)
    = true ->
    (∀ x, x ∈ params -> var_ok x) ->
    ∃ (sh : list pshape) (pk pk' : list (variable * flambda_kind)),
      filter_kept (map is_kept pd_cc) params = map fst pk
      ∧ flat_map (fun '(x, d) => map fst (decision_params x
                                             (param_decision_at unboxed x d)))
          (combine params pd)
        = map fst pk'
      ∧ params_rel2 unboxed dead sh pk pk'
      ∧ Forall2 fits (flat_map pshape_of (pdescs_of pd (map is_kept pd_cc)
                                           kinds))
          sh.
Proof.
  induction params as [ | x params IH]; intros pd pd_cc kinds Hlink Hlen Hok
    Hpok Hvars.
  - destruct (pdescs_of pd (map is_kept pd_cc) kinds) eqn:E;
      [ | discriminate Hlen ].
    exists [], [], []. split; [ destruct (map is_kept pd_cc); reflexivity | ].
    split; [ destruct pd; reflexivity | ].
    split; constructor.
  - destruct pd as [ | d pd]; [ discriminate Hlen | ].
    destruct pd_cc as [ | dcc pd_cc]; [ inversion Hlink | ].
    destruct kinds as [ | kind kinds]; [ discriminate Hlen | ].
    inversion Hlink as [ | d0 dcc0 pd0 pd_cc0 Hl Hlink']; subst.
    cbn [map pdescs_of] in Hlen, Hok, Hpok |- *.
    simpl in Hlen. injection Hlen as Hlen.
    cbn [forallb] in Hok. apply andb_true_iff in Hok as [Hp Hok].
    cbn [forallb2] in Hpok. apply andb_true_iff in Hpok as [Hx Hpok].
    destruct (IH pd pd_cc kinds Hlink' Hlen Hok Hpok
                (fun y Hy => Hvars y (or_intror Hy)))
      as (sh & pk & pk' & Hpk & Hpk' & Hrel & Hfits).
    destruct (Hvars x (or_introl eq_refl)) as [Hfresh Hdead].
    unfold pdesc_ok in Hp. cbn [pd_kind pd_decision pd_keep] in Hp.
    apply andb_true_iff in Hp as [Hkind Hp].
    unfold code_param_ok in Hx. cbn [pd_decision] in Hx.
    cbn [combine flat_map].
    destruct d as [dd | fields | kk]; [ | | contradiction ].
    + cbn in Hl. subst dcc. cbn [param_decision_at decision_params].
      destruct dd as [kk | ].
      * cbn [is_kept filter_kept]. apply andb_true_iff in Hp as [_ Hk].
        apply kind_eqb_eq in Hk. subst kk.
        simpl in Hx.
        assert (Hxn : fields_of unboxed x = None).
        { unfold FlambdaReaperUnboxingRefinesSpec.plain_variable in Hx.
          apply negb_true_iff in Hx.
          unfold Keyed.is_bound in Hx. unfold fields_of.
          destruct (Keyed.lookup _ _ _); [ discriminate Hx | reflexivity ]. }
        destruct (s_has_usage sol (Node_variable x)) eqn:Hu.
        -- exists (Psh_plain :: sh), ((x, kind) :: pk), ((x, kind) :: pk').
           split; [ cbn [map]; f_equal; exact Hpk | ].
           split; [ cbn [map app]; f_equal; exact Hpk' | ].
           split; [ apply params_rel2_plain; [ | exact Hrel ] | ].
           ++ apply (plain2_of x kind Hxn Hfresh Hkind).
              unfold is_var_used. destruct kind; try discriminate Hkind;
                exact Hu.
           ++ cbn [flat_map pshape_of pd_keep pd_decision app].
              constructor; [ constructor | exact Hfits ].
        -- exists (Psh_dead kind :: sh), ((x, kind) :: pk), ((x, kind) :: pk').
           split; [ cbn [map]; f_equal; exact Hpk | ].
           split; [ cbn [map app]; f_equal; exact Hpk' | ].
           split; [ apply params_rel2_dead;
                    [ exact (Hdead eq_refl Hxn) | exact Hxn | exact Hfresh
                    | exact Hrel ] | ].
           cbn [flat_map pshape_of pd_keep pd_decision app].
           constructor; [ constructor | exact Hfits ].
      * cbn [is_kept filter_kept].
        exists sh, pk, pk'. split; [ exact Hpk | ].
        split; [ exact Hpk' | split; [ exact Hrel | ] ].
        cbn [flat_map pshape_of pd_keep app]. exact Hfits.
    + destruct (fields_of unboxed x) as [fields_x | ] eqn:Hfx;
        [ | discriminate Hx ].
      apply same_paths_eq in Hx.
      cbn [param_decision_at].
      change (unboxed_of_node unboxed (Node_variable x))
        with (fields_of unboxed x).
      rewrite Hfx. cbn [decision_params].
      apply andb_true_iff in Hp as [Hv Hkeep]. apply kind_eqb_eq in Hv.
      subst kind.
      destruct (is_kept dcc) eqn:Hk; cbn [filter_kept].
      * exists (Psh_unboxed fields_x :: sh), ((x, Value) :: pk),
          (fields_leaves fields_x ++ pk').
        split; [ cbn [map]; f_equal; exact Hpk | ].
        split; [ rewrite map_app, Hpk'; reflexivity | ].
        split; [ apply params_rel2_unboxed; assumption | ].
        cbn [flat_map pshape_of pd_keep pd_decision app].
        constructor; [ constructor; exact Hx | exact Hfits ].
      * cbn [orb] in Hkeep.
        apply no_leaves_paths in Hkeep. rewrite Hx in Hkeep.
        exists sh, pk, pk'. split; [ exact Hpk | ].
        split; [ rewrite (leaves_of_paths_nil fields_x Hkeep); exact Hpk' | ].
        split; [ exact Hrel | ].
        cbn [flat_map pshape_of pd_keep app]. exact Hfits.
Qed.

Lemma group_by_sizes_length {A : Type} (sizes : list nat) :
  ∀ l : list A, length (group_by_sizes sizes l) = length sizes.
Proof.
  induction sizes as [ | n sizes IH]; intros l; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma group_by_parameter_length {A : Type} (a : arity) (l : list A) :
  length (group_by_parameter a l) = length a.
Proof.
  unfold group_by_parameter. rewrite group_by_sizes_length.
  unfold unarize_per_parameter. rewrite !length_map. reflexivity.
Qed.

Lemma prepend_first_length {A : Type} (l : list A) (gs : list (list A)) :
  length (prepend_first l gs) = length gs.
Proof. destruct gs; reflexivity. Qed.

(* The arguments of a call of code that changes its calling convention,
   of one group. *)
Lemma call_args_rel (en : rebuild_env) (enu : unbox_env) (args : list simple) :
  env_kinds (ue_env enu) = env_kinds en ->
  (∀ x, kind_ok (kind_of_variable en x) = true) ->
  (∀ z, z ∈ variables_in_simples args -> fresh z) ->
  ∀ (pd : list decision) (pd_cc : list param_decision)
    (kinds : list flambda_kind),
    Forall2 code_link pd pd_cc ->
    length args = length (pdescs_of pd (map is_kept pd_cc) kinds) ->
    length pd = length kinds -> length pd_cc = length kinds ->
    forallb pdesc_ok (pdescs_of pd (map is_kept pd_cc) kinds) = true ->
    args_ok2 sol unboxed (env_kinds en) (pdescs_of pd (map is_kept pd_cc) kinds)
      args
    = true ->
    ∃ l_u : list (simple * flambda_kind),
      get_args_with_kinds_u sol unboxed (ue_env enu) pd args = Some l_u
      ∧ args_rel2 unboxed dead
          (flat_map pshape_of (pdescs_of pd (map is_kept pd_cc) kinds))
          (map fst (get_args_with_kinds sol en pd_cc args)) (map fst l_u)
      ∧ map snd l_u = decisions_kinds pd
      ∧ shape_kinds
          (flat_map pshape_of (pdescs_of pd (map is_kept pd_cc) kinds))
          (map snd (get_args_with_kinds sol en pd_cc args))
        = decisions_kinds pd.
Proof.
  intros Hkinds Hkok. induction args as [ | a args IH];
    intros Hfresh pd pd_cc kinds Hlink Hlen Hl1 Hl2 Hok Hargs.
  - rewrite pdescs_of_length, length_map in Hlen. simpl in Hlen.
    assert (Hpd0 : pd = []).
    { destruct pd; [ reflexivity | ]. exfalso.
      destruct pd_cc; destruct kinds; simpl in Hl1, Hl2, Hlen;
        try discriminate Hl1; try discriminate Hl2; discriminate Hlen. }
    subst pd. inversion Hlink; subst.
    exists []. split; [ reflexivity | ].
    split; [ apply args_rel2_beyond; reflexivity | ].
    split; reflexivity.
  - destruct pd as [ | d pd]; [ discriminate Hlen | ].
    destruct pd_cc as [ | dcc pd_cc]; [ inversion Hlink | ].
    destruct kinds as [ | kind kinds]; [ discriminate Hlen | ].
    inversion Hlink as [ | d0 dcc0 pd0 pd_cc0 Hl Hlink']; subst.
    cbn [map pdescs_of] in Hlen, Hok, Hargs |- *.
    simpl in Hlen, Hl1, Hl2. injection Hlen as Hlen.
    injection Hl1 as Hl1. injection Hl2 as Hl2.
    cbn [forallb] in Hok. apply andb_true_iff in Hok as [Hp Hok].
    cbn [args_ok2] in Hargs. apply andb_true_iff in Hargs as [Ha Hargs].
    unfold variables_in_simples in Hfresh. cbn [flat_map] in Hfresh.
    assert (Ha_fresh : ∀ z, z ∈ variables_in_simple a -> fresh z)
      by (intros z Hz; apply Hfresh, in_or_app; left; exact Hz).
    destruct (IH (fun z Hz => Hfresh z (in_or_app _ _ _ (or_intror Hz)))
                pd pd_cc kinds Hlink' Hlen Hl1 Hl2 Hok Hargs)
      as (rest & Hrest & Hrel & Hk1 & Hk2).
    cbn [get_args_with_kinds_u]. rewrite Hrest.
    unfold get_args_with_kinds in *. cbn [combine flat_map].
    unfold decisions_kinds in *. cbn [flat_map].
    unfold pdesc_ok in Hp. cbn [pd_kind pd_decision pd_keep] in Hp.
    apply andb_true_iff in Hp as [Hkind Hp].
    unfold arg_ok in Ha. cbn [pd_decision pd_keep] in Ha.
    destruct d as [dd | fields | kk]; [ | | contradiction ].
    + cbn in Hl. subst dcc. destruct dd as [kk | ].
      * pose proof (strong_simple2 a Ha Ha_fresh) as Hstrong.
        exists ((rewrite_simple sol_u (ue_env enu) a, kk) :: rest).
        split; [ reflexivity | ].
        rewrite (rewrite_simple_kinds sol_u (ue_env enu) en a Hkinds),
          (rewrite_simple_u sol unboxed en a Hstrong).
        cbn [map app fst snd flat_map pshape_of pd_keep pd_decision is_kept].
        split; [ apply args_rel2_plain; [ apply rewrite_plain2; assumption
                                        | exact Hrel ] | ].
        unfold decision_kinds. cbn [decision_params map app].
        split; [ f_equal; exact Hk1 | ].
        cbn [shape_kinds]. f_equal. exact Hk2.
      * exists rest. split; [ reflexivity | ].
        cbn [map app fst snd flat_map pshape_of pd_keep is_kept].
        unfold decision_kinds. cbn [decision_params map app].
        split; [ exact Hrel | split; [ exact Hk1 | exact Hk2 ] ].
    + unfold unboxed_arg_ok in Ha.
      destruct a as [y | sym | c]; try discriminate Ha.
      destruct (fields_of unboxed y) as [fields_y | ] eqn:Hy;
        [ | discriminate Ha ].
      apply andb_true_iff in Ha as [Halike Hused].
      destruct (unboxed_args_pairs unboxed fields y fields_y Hy Halike)
        as (pairs & Hpairs & Hlu).
      rewrite Hlu. cbn [option_map].
      exists (map (fun '(_, x, kind) => (Var x, kind)) pairs ++ rest).
      split; [ reflexivity | ].
      destruct (pairs_spec (Unboxed fields) Value (Unboxed fields_y) pairs
                  Hpairs)
        as (_ & _ & _ & Hthirds).
      assert (Hpk : map snd (map (fun '(_, x, kind) => (Var x, kind)) pairs)
                    = map snd (fields_leaves fields)).
      { rewrite map_map. unfold fields_leaves. rewrite <- Hthirds.
        apply map_ext. intros [[x x'] k']. reflexivity. }
      assert (Hpf : map fst (map (fun '(_, x, kind) => (Var x, kind)) pairs)
                    = map (fun '(_, x', _) => Var x') pairs).
      { rewrite map_map. apply map_ext. intros [[x x'] k']. reflexivity. }
      apply andb_true_iff in Hp as [_ Hkeep].
      destruct (is_kept dcc) eqn:Hk.
      * destruct dcc as [kcc | ]; [ | discriminate Hk ].
        cbn [app map fst snd flat_map pshape_of pd_keep pd_decision].
        simpl in Hused. unfold kinds_env in Hused.
        cbn [rewrite_simple]. unfold kind_of_variable in Hused |- *.
        cbn [env_kinds] in Hused. rewrite Hused.
        rewrite map_app, Hpf.
        split; [ exact (args_rel2_unboxed unboxed dead _ fields fields_y y
                          pairs _ _ Hy Hpairs Hrel) | ].
        split; [ rewrite map_app, Hpk; unfold decision_kinds;
                 cbn [decision_params]; f_equal; exact Hk1 | ].
        cbn [shape_kinds]. unfold decision_kinds.
        cbn [decision_params]. rewrite leaves_kinds. f_equal. exact Hk2.
      * destruct dcc as [kcc | ]; [ discriminate Hk | ].
        cbn [app map fst snd flat_map pshape_of pd_keep].
        cbn [orb] in Hkeep.
        rewrite (pairs_no_leaves fields fields_y pairs Hpairs
                   (no_leaves_nil fields Hkeep)).
        cbn [map app].
        split; [ exact Hrel | ].
        unfold decision_kinds. cbn [decision_params].
        rewrite (no_leaves_nil fields Hkeep). cbn [map app].
        split; [ exact Hk1 | exact Hk2 ].
Qed.

Lemma rewrite_call_kind_direct (ap : apply_expr) (id : code_id) :
  code_id_actually_called F sol ap = Some id ->
  rewrite_call_kind F sol pdc unit_code_ids ap = Direct id.
Proof.
  unfold code_id_actually_called, rewrite_call_kind, called_call_kind.
  destruct (ap_call_kind ap); [ | intros H; discriminate H | ];
    destruct (ap_callee ap); try (intros H; discriminate H);
    destruct (code_id_actually_directly_called F sol _) as [ids | ];
    try (intros H; discriminate H); intros Hs; rewrite Hs; reflexivity.
Qed.

(* The arm of a direct call, or one of known arity. *)
Lemma known_kind_match {A : Type} (kind : call_kind) (x y : A) :
  kind <> Indirect_unknown_arity ->
  match kind with
  | Direct _ | Indirect_known_arity _ => x
  | Indirect_unknown_arity => y
  end = x.
Proof.
  destruct kind; [ reflexivity | intros H; exfalso; apply H; reflexivity
                 | reflexivity ].
Qed.

Lemma simple_is_dead_kinds (en1 en2 : rebuild_env) (s : simple) :
  env_kinds en1 = env_kinds en2 ->
  simple_is_dead sol en1 s = simple_is_dead sol en2 s.
Proof.
  intros H. destruct s; simpl; [ | reflexivity | reflexivity ].
  unfold kind_of_variable. rewrite H. reflexivity.
Qed.

Lemma no_dead_code (pd : list decision) (pd_cc : list param_decision)
    (keeps : list bool) (kinds : list flambda_kind) :
  Forall2 code_link pd pd_cc ->
  no_dead (flat_map pshape_of (pdescs_of pd keeps kinds)).
Proof.
  intros Hl. revert keeps kinds.
  induction Hl as [ | d dcc pd pd_cc Hd _ IH]; intros keeps kinds kind Hin;
    [ destruct keeps; destruct kinds; destruct Hin | ].
  destruct keeps as [ | b keeps]; [ destruct Hin | ].
  destruct kinds as [ | k kinds]; [ destruct Hin | ].
  cbn [pdescs_of flat_map] in Hin. apply in_app_or in Hin as [Hin | Hin].
  - unfold pshape_of in Hin. cbn [pd_keep pd_decision] in Hin.
    destruct b; [ | destruct Hin ].
    destruct Hin as [Heq | []]. destruct d; try discriminate Heq.
    contradiction.
  - exact (IH keeps kinds kind Hin).
Qed.

(* The variables of the fields of a result that the callee unboxes, which a
   wrapper takes as parameters. *)
Lemma freshen_unbox (fr : unboxed_fields) (kind : flambda_kind)
    (n : positive) :
  ∃ fr' : unboxed_fields,
    freshen_decisions [(Unbox fr, kind)] n = [(Wrapper_unbox fr', kind)]
    ∧ fields_paths fr' = fields_paths fr
    ∧ NoDup (leaf_variables fr')
    ∧ fst (rename_unboxed_field (Unboxed fr) n) = Unboxed fr'.
Proof.
  destruct (rename_ok (Unboxed fr) Value n)
    as ((fr' & Hshape) & Hpaths & Hnd & _ & _).
  pose proof (Hshape fr eq_refl) as Hfr'.
  exists fr'. cbn [freshen_decisions].
  destruct (rename_unboxed_field (Unboxed fr) n) as [u n'] eqn:Hr.
  cbn [fst] in Hfr', Hpaths, Hnd. subst u.
  split; [ reflexivity | split; [ exact Hpaths | split; [ exact Hnd | ] ] ].
  reflexivity.
Qed.

(* Values of the shape of one unboxed result. *)
Lemma vrel_one_unboxed (fr : unboxed_fields) (vs vs' : list value)
    (kind : flambda_kind) :
  vrel [Psh_unboxed fr] vs vs' -> map kind_of_value vs = [kind] ->
  ∃ v : value, vs = [v] ∧ fields_vals fr v vs'.
Proof.
  intros Hv Hk. inversion Hv; subst; [ discriminate Hk | ].
  match goal with
  | H : vrel [] ?a ?b |- _ => inversion H; subst
  end.
  match goal with
  | H : map kind_of_value (?v :: ?l) = [_] |- _ =>
    destruct l; [ | discriminate H ]; exists v
  end.
  rewrite app_nil_r. split; [ reflexivity | assumption ].
Qed.

Lemma vrel_one_plain (vs vs' : list value) (kind : flambda_kind) :
  vrel [Psh_plain] vs vs' -> map kind_of_value vs = [kind] ->
  ∃ v : value, vs = [v] ∧ vs' = [v].
Proof.
  intros Hv Hk. inversion Hv; subst; [ discriminate Hk | ].
  match goal with
  | H : vrel [] ?a ?b |- _ => inversion H; subst
  end.
  match goal with
  | H : map kind_of_value (?v :: ?l) = [_] |- _ =>
    destruct l; [ | discriminate H ]; exists v
  end.
  split; reflexivity.
Qed.

Lemma vrel_none (vs vs' : list value) :
  vrel [] vs vs' -> vs' = vs.
Proof. intros H. inversion H. reflexivity. Qed.

Lemma unbox_kinds (fr fr' : unboxed_fields) (v : value) (ws : list value) :
  fields_paths fr' = fields_paths fr -> fields_vals fr v ws ->
  map kind_of_value ws = map snd (fields_leaves fr').
Proof.
  intros Hp Hws. rewrite (fields_vals_kinds fr v ws Hws), leaves_kinds, Hp.
  reflexivity.
Qed.



Lemma cc_rebuild_code (id : code_id) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (body : expr) :
  rebuild_code_cc F sol cannot_change code_deps pdc unit_code_ids codes ff id
    (Code params mc ret exn md body)
  = let change :=
      code_calling_convention_change sol cannot_change code_deps
        unit_code_ids id md
    in
    let bound :=
      (mc, Value) :: with_kinds params (unarize (cm_params_arity md))
    in
    let en :=
      match change with
      | Not_changing_calling_convention => empty_env
      | Changing_calling_convention _ return_decisions =>
        bind_continuations empty_env [(ret, map is_kept return_decisions)]
      end
    in
    let body :=
      if existsb (fun '(x, kind) => is_dead_var sol x kind) bound
      then Invalid
      else cc_rebuild (bind_variables en bound) body
    in
    match change with
    | Not_changing_calling_convention =>
      Code params mc ret exn (erase_code_subkinds md) body
    | Changing_calling_convention params_decisions return_decisions =>
      Code (filter_kept (map is_kept params_decisions) params) mc ret exn
        (changed_code_metadata md params_decisions return_decisions) body
    end.
Proof. reflexivity. Qed.

Lemma u_rebuild_code (id : code_id) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (body : expr) :
  rebuild_code_u true F sol cannot_change code_deps pdc unit_code_ids codes
    unboxed ffu [] id (Code params mc ret exn md body)
  = let change :=
      code_change_u sol cannot_change code_deps unit_code_ids unboxed id md
    in
    let bound :=
      (mc, Value) :: with_kinds params (unarize (cm_params_arity md))
    in
    let en :=
      ubind_continuations empty_unbox_env
        (code_continuation_decisions sol code_deps unboxed id md change ret
           exn)
    in
    let body :=
      if existsb (fun '(x, kind) => is_dead_var sol x kind) bound
      then Invalid
      else u_rebuild (ubind_variables en bound) body
    in
    match change with
    | Not_changing_u => Code params mc ret exn (erase_code_subkinds md) body
    | Changing_u params_decisions return_decisions =>
      let my_closure_decision :=
        match
          my_closure_decision_u cannot_change code_deps unit_code_ids unboxed
            id
        with
        | Some fields => param_decision_at unboxed mc (Unbox fields)
        | None => Decided Delete
        end
      in
      Code
        (map fst
           (decision_params mc my_closure_decision
            ++ flat_map
                 (fun '(x, d) =>
                    decision_params x (param_decision_at unboxed x d))
                 (combine params params_decisions)))
        mc ret exn
        (changed_code_metadata_u md my_closure_decision params_decisions
           return_decisions)
        body
    end.
Proof. reflexivity. Qed.

Lemma flat_map_decision_params (params : list variable) (pd : list decision) :
  map fst
    (flat_map
       (fun '(x, d) => decision_params x (param_decision_at unboxed x d))
       (combine params pd))
  = flat_map
      (fun '(x, d) =>
         map fst (decision_params x (param_decision_at unboxed x d)))
      (combine params pd).
Proof.
  induction (combine params pd) as [ | [x d] l IH]; [ reflexivity | ].
  cbn [flat_map]. rewrite map_app, IH. reflexivity.
Qed.

Lemma pdescs_of_decisions (pd : list decision) (keeps : list bool)
    (kinds : list flambda_kind) :
  length pd = length kinds -> length keeps = length kinds ->
  map pd_decision (pdescs_of pd keeps kinds) = pd
  ∧ map pd_keep (pdescs_of pd keeps kinds) = keeps.
Proof.
  revert keeps kinds. induction pd as [ | d pd IH]; intros [ | b keeps]
    [ | k kinds] H1 H2; simpl in H1, H2 |- *; try discriminate H1;
    try discriminate H2; [ split; reflexivity | ].
  destruct (IH keeps kinds ltac:(lia) ltac:(lia)) as [Ha Hb].
  rewrite Ha, Hb. split; reflexivity.
Qed.

Lemma decisions_u_length (nodes : list node) (kinds : list flambda_kind) :
  length (decisions_u sol unboxed nodes kinds)
  = length (combine nodes kinds).
Proof. unfold decisions_u. apply length_map. Qed.

Lemma decisions_of_length' (nodes : list node) (kinds : list flambda_kind) :
  length (decisions_of sol nodes kinds) = length (combine nodes kinds).
Proof. unfold decisions_of. apply length_map. Qed.

Lemma with_kinds_kinds_ok (params : list variable) (kinds : list flambda_kind)
    (x : variable) (kind : flambda_kind) :
  (∀ k, k ∈ kinds -> kind_ok k = true) ->
  (x, kind) ∈ with_kinds params kinds -> kind_ok kind = true.
Proof.
  revert kinds. induction params as [ | p params IH]; intros [ | k kinds] Hk
    Hin; simpl in Hin; [ destruct Hin | destruct Hin | | ].
  - destruct Hin as [Heq | Hin]; [ injection Heq as _ <-; reflexivity | ].
    exact (IH [] Hk Hin).
  - destruct Hin as [Heq | Hin];
      [ injection Heq as _ <-; apply Hk; left; reflexivity | ].
    apply (IH kinds); [ | exact Hin ]. intros k' Hk'. apply Hk. right.
    exact Hk'.
Qed.

Lemma pdescs_kinds_ok (pd : list decision) (keeps : list bool)
    (kinds : list flambda_kind) :
  length (pdescs_of pd keeps kinds) = length kinds ->
  forallb pdesc_ok (pdescs_of pd keeps kinds) = true ->
  ∀ k, k ∈ kinds -> kind_ok k = true.
Proof.
  revert keeps kinds. induction pd as [ | d pd IH]; intros [ | b keeps]
    [ | k0 kinds] Hlen Hok k Hk; simpl in Hlen, Hok; try discriminate Hlen;
    try (destruct Hk; fail).
  apply andb_true_iff in Hok as [Hp Hok].
  destruct Hk as [<- | Hk].
  - unfold pdesc_ok in Hp. apply andb_true_iff in Hp as [Hp _]. exact Hp.
  - exact (IH keeps kinds ltac:(lia) Hok k Hk).
Qed.

(* The parameters of code that keeps its calling convention, as the
   decisions of code that changes it would have them. *)
Lemma nc_pdescs (kinds : list flambda_kind) :
  pdescs_of (map (fun k => Decided (Keep k)) kinds)
    (map is_kept (map Keep kinds)) kinds
  = map (fun kind => Pdesc (Decided (Keep kind)) true kind) kinds.
Proof.
  induction kinds as [ | k kinds IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma nc_link (kinds : list flambda_kind) :
  Forall2 code_link (map (fun k => Decided (Keep k)) kinds) (map Keep kinds).
Proof.
  induction kinds as [ | k kinds IH]; constructor; [ reflexivity | exact IH ].
Qed.

Lemma nc_mask_all (kinds : list flambda_kind) :
  forallb (fun b => b) (map is_kept (map Keep kinds)) = true.
Proof. induction kinds as [ | k kinds IH]; [ reflexivity | exact IH ]. Qed.

Lemma nc_params (params : list variable) :
  ∀ kinds : list flambda_kind,
    length params = length kinds ->
    flat_map
      (fun '(x, d) =>
         map fst (decision_params x (param_decision_at unboxed x d)))
      (combine params (map (fun k => Decided (Keep k)) kinds))
    = params.
Proof.
  induction params as [ | x params IH]; intros [ | k kinds] H;
    simpl in H |- *; try discriminate H; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma all_kept_pdescs (rd : list decision) :
  ∀ kinds : list flambda_kind,
    forallb pd_keep (pdescs_of rd (map (fun _ => true) rd) kinds) = true.
Proof.
  induction rd as [ | d rd IH]; intros [ | k kinds]; simpl; try reflexivity.
  exact (IH kinds).
Qed.

(* Code of the unit that keeps its calling convention, as the two rebuilds
   make it: the same parameters, and its return continuation with no mask
   in the first and the decisions of fixed arity in the second. *)
(* The return continuation of code taken to never return is not free in the
   second body: it is not live in the body ([calls_code_f3]). *)
Lemma never_ret_dead (id : code_id) (ret : continuation) (body q : expr) :
  negb (nr id)
  || negb (among ret (live_conts F sol code_deps codes body)) = true ->
  incl (free_continuations_of_expr q) (live_conts F sol code_deps codes body)
  -> NRf id = true -> ret ∉ free_continuations_of_expr q.
Proof.
  intros Hok Hincl Hnr Hin. unfold NRf in Hnr. rewrite Hnr in Hok.
  apply Hincl, among_In in Hin. rewrite Hin in Hok. discriminate Hok.
Qed.

(* The body of code taken to never return, whose return continuation is
   not free in the second, is related at [never_ksh]. *)
Lemma code_K_of_body (id : code_id) (c c' : code) (cs : code_shape) :
  uc unboxed dead CSf NRf RCHf OVf REMf
      (code_K c cs) (code_body c) (code_body c') ->
  (NRf id = true ->
   code_return_continuation c ∉ free_continuations_of_expr (code_body c')) ->
  uc unboxed dead CSf NRf RCHf OVf REMf (code_K_of NRf id c cs) (code_body c)
    (code_body c').
Proof.
  intros Huc Hnr. unfold code_K_of. destruct (NRf id); [ | exact Huc ].
  unfold code_K_never. unfold code_K in Huc.
  exact (uc_never_shadow _ _ _ _ _ _ _ _ _ _ _ _ Huc (Hnr eq_refl)).
Qed.

Lemma code_rebuilt_nc (id : code_id) (c : code) :
  rebuild_ok_expr (code_body c) ->
  calls_code_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
    nr reach ovr rem id
    c
  = true ->
  (∀ z, z ∈ variables_in_code c -> var_ok z) ->
  (∀ k, k ∈ continuations_in_code c -> k <> ff ∧ k <> ffu) ->
  FlambdaReaperFlowProofs.code_edges (graph_of_facts F) noted deps cds
    kept_slots codes0 symbols True False id c ->
  code_change_u sol cannot_change code_deps unit_code_ids unboxed id
    (metadata_of_code c)
  = Not_changing_u ->
  let ccc :=
    rebuild_code_cc F sol cannot_change code_deps pdc unit_code_ids codes ff
      id c
  in
  let cu :=
    rebuild_code_u true F sol cannot_change code_deps pdc unit_code_ids
      codes unboxed ffu [] id c
  in
  (∃ cs : code_shape,
     code_frame unboxed dead CSf id ccc cu cs
     ∧ uc unboxed dead CSf NRf RCHf OVf REMf (code_K_of NRf id ccc cs)
         (code_body ccc) (code_body cu))
  ∧ incl (free_continuations_of_code cu) (free_continuations_of_code ccc)
  ∧ incl (free_continuations_of_code cu)
      (live_conts_code F sol code_deps codes c).
Proof.
  destruct c as [params mc ret exn md body].
  intros IH Hok Hvars Hconts Hedges Hnc. cbv zeta.
  cbn [metadata_of_code] in Hnc.
  cbn [calls_code_f3] in Hok. apply andb_true_iff in Hok as [Hok Hnrb].
  destruct (lookup id codes) as [md' | ] eqn:Hcodes;
    [ | discriminate Hok ].
  apply andb_true_iff in Hok as [Hmd Hok].
  apply code_metadata_eqb_eq in Hmd. subst md'.
  assert (Hncc : code_calling_convention_change sol cannot_change code_deps
                   unit_code_ids id md
                 = Not_changing_calling_convention).
  { unfold code_change_u in Hnc. unfold code_calling_convention_change.
    destruct (cannot_change_calling_convention cannot_change unit_code_ids
                id);
      [ reflexivity | ].
    destruct (lookup id code_deps); [ discriminate Hnc | reflexivity ]. }
  destruct (code_pdescs sol cannot_change code_deps unit_code_ids unboxed id
              md)
    as [[pds rds] | ] eqn:Hpd; [ | discriminate Hok ].
  pose proof Hpd as Hpd'.
  unfold code_pdescs in Hpd. rewrite Hnc, Hncc in Hpd.
  destruct (lookup id code_deps) as [cd | ] eqn:Hcd; [ | discriminate Hpd ].
  set (rd := fixed_arity_decisions sol unboxed (cd_return cd)
               (unarize (cm_result_arity md))) in *.
  destruct (cm_is_tupled md || existsb is_unbox rd) eqn:Htup;
    [ discriminate Hpd | ].
  apply orb_false_iff in Htup as [Htup _].
  injection Hpd as Hpds Hrds.
  apply andb_true_iff in Hok as [Hok Hcalls].
  destruct (code_exn_decision sol code_deps unboxed id) as [de | ] eqn:Hde;
    [ | discriminate Hcalls ].
  destruct (code_exn_decision_cases id de Hde) as (Hdc & cd0 & Hcd0 & Hfad).
  apply andb_true_iff in Hok as [Hok Hre].
  apply andb_true_iff in Hok as [Hok Hmc].
  apply andb_true_iff in Hok as [Hok Hmcd].
  apply andb_true_iff in Hok as [Hok Hpok].
  apply andb_true_iff in Hok as [Hok Hrok].
  apply andb_true_iff in Hok as [Hok Hdok].
  apply andb_true_iff in Hok as [Hok Hrlen].
  apply andb_true_iff in Hok as [Hlen Hdlen].
  apply Nat.eqb_eq in Hlen, Hdlen, Hrlen.
  apply negb_true_iff in Hre. apply Pos.eqb_neq in Hre.
  set (kinds_p := unarize (cm_params_arity md)) in *.
  set (kinds_r := unarize (cm_result_arity md)) in *.
  assert (Hlrd : length rd = length kinds_r).
  { rewrite <- Hrds, pdescs_of_length, length_map in Hrlen.
    assert (Hle : length rd <= length kinds_r).
    { unfold rd, fixed_arity_decisions. rewrite length_map, length_combine.
      lia. }
    lia. }
  destruct (pdescs_of_decisions rd (map (fun _ => true) rd) kinds_r Hlrd
              ltac:(rewrite length_map; exact Hlrd))
    as [Hrdec _].
  rewrite Hrds in Hrdec.
  destruct Hedges as (cd' & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _
                      & Hbody_edges).
  rewrite cc_rebuild_code, u_rebuild_code. cbv zeta.
  rewrite Hncc, Hnc. fold kinds_p kinds_r.
  assert (Hccd : code_continuation_decisions sol code_deps unboxed id md
                   Not_changing_u ret exn
                 = [(ret, rd);
                    (exn, fixed_arity_decisions sol unboxed [cd_exn cd]
                            [Value])])
    by (unfold code_continuation_decisions; rewrite Hcd; reflexivity).
  rewrite Hccd.
  set (bound := (mc, Value) :: with_kinds params kinds_p).
  assert (Hpds_ok : ∀ k, k ∈ kinds_p -> kind_ok k = true).
  { intros k Hk. rewrite <- Hpds in Hdok. rewrite forallb_forall in Hdok.
    assert (Hin : Pdesc (Decided (Keep k)) true k
                  ∈ map (fun kind => Pdesc (Decided (Keep kind)) true kind)
                      kinds_p)
      by (apply in_map_iff; exists k; split; [ reflexivity | exact Hk ]).
    specialize (Hdok _ Hin). unfold pdesc_ok in Hdok.
    apply andb_true_iff in Hdok as [Hdok _]. exact Hdok. }
  assert (Hbound_ok : ∀ y kind, (y, kind) ∈ bound -> kind_ok kind = true).
  { intros y kind [Heq | Hin]; [ injection Heq as _ <-; reflexivity | ].
    exact (with_kinds_kinds_ok params kinds_p y kind Hpds_ok Hin). }
  assert (Hcd0' : cd0 = cd) by congruence. subst cd0.
  assert (Hci : cinv [(ret, (rds, true)); (exn, ([Pdesc de true Value], false))]
                  (bind_variables empty_env bound)
                  (ubind_variables
                     (ubind_continuations empty_unbox_env
                        [(ret, rd);
                         (exn, fixed_arity_decisions sol unboxed [cd_exn cd]
                                 [Value])])
                     bound)).
  { split.
    - intros k ps h Hl. rewrite !lookup_cons in Hl.
      destruct (Pos.eqb k ret); [ injection Hl as <- <-; exact Hrok | ].
      destruct (Pos.eqb k exn); [ | discriminate Hl ].
      injection Hl as <- <-. destruct Hdc as [-> | ->]; reflexivity.
    - reflexivity.
    - intros x. apply kind_of_bind; [ intros y; reflexivity | ].
      exact Hbound_ok.
    - reflexivity.
    - reflexivity.
    - reflexivity.
    - intros k. cbn [ue_decisions ubind_variables ubind_variables_then
                     ubind_set ubind_continuations empty_unbox_env app].
      rewrite !lookup_cons. destruct (Pos.eqb k ret) eqn:Hkr;
        [ cbn; rewrite Hrdec; reflexivity | ].
      destruct (Pos.eqb k exn) eqn:Hke; [ | reflexivity ].
      rewrite Hfad. reflexivity.
    - intros k. cbn [env_conts bind_variables bind_variables_then
                     empty_env].
      rewrite !lookup_cons. destruct (Pos.eqb k ret) eqn:Hkr.
      + right. split; [ reflexivity | ]. exists rds, true.
        split; [ reflexivity | ]. rewrite <- Hrds. apply all_kept_pdescs.
      + destruct (Pos.eqb k exn) eqn:Hke; [ | left; reflexivity ].
        right. split; [ reflexivity | ].
        exists [Pdesc de true Value], false. split; reflexivity. }
  assert (Hcs : CSf id
                = Some
                    (Code_shape (flat_map pshape_of pds)
                       (flat_map pshape_of rds) (flat_map kept_kind rds)
                       (cm_params_arity md) (cm_result_arity md)
                       (pshape1 (Pdesc de true Value)) true)).
  { unfold CSf. rewrite Hcodes, Hde, Hnc, Hpd'. reflexivity. }
  rewrite <- Hpds in Hlen, Hdok, Hpok, Hcs.
  rewrite <- (nc_pdescs kinds_p) in Hlen, Hdok, Hpok, Hcs.
  destruct (code_params_rel2 params (map (fun k => Decided (Keep k)) kinds_p)
              (map Keep kinds_p) kinds_p (nc_link kinds_p) Hlen Hdok Hpok)
    as (sh & pk & pk' & Hpk & Hpk' & Hrel & Hfits).
  { intros x Hx. apply Hvars. right. apply in_or_app. left. exact Hx. }
  rewrite (filter_kept_all params _ (nc_mask_all kinds_p)) in Hpk.
  rewrite nc_params in Hpk'
    by (rewrite Hlen, pdescs_of_length, !length_map; lia).
  assert (Hmcf : fields_of unboxed mc = None).
  { cbn in Hmc. apply negb_true_iff in Hmc. unfold Keyed.is_bound in Hmc.
    unfold fields_of. destruct (Keyed.lookup _ _ _);
      [ discriminate Hmc | reflexivity ]. }
  assert (Hframe : ∀ body body' : expr,
            (binds_statics body' = true -> binds_statics body = true) ->
            code_frame unboxed dead CSf id
              (Code params mc ret exn (erase_code_subkinds md) body)
              (Code params mc ret exn (erase_code_subkinds md) body')
              (Code_shape
                 (flat_map pshape_of
                    (pdescs_of (map (fun k => Decided (Keep k)) kinds_p)
                       (map is_kept (map Keep kinds_p)) kinds_p))
                 (flat_map pshape_of rds) (flat_map kept_kind rds)
                 (cm_params_arity md) (cm_result_arity md)
                 (pshape1 (Pdesc de true Value)) true)).
  { intros b b' Hbs. split; [ exact Hcs | ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | ].
    split; [ intros _; split; reflexivity | ].
    split; [ exact Htup | split; [ exact Htup | ] ].
    split; [ reflexivity | split; [ exact Hmcf | ] ].
    split; [ exact (proj1 (Hvars mc (or_introl eq_refl))) | ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ exact Hbs | ].
    split; [ apply erased_subkinds_say_nothing | ].
    exists sh, pk, pk'. split; [ exact Hpk | ].
    split; [ exact Hpk' | split; [ exact Hrel | exact Hfits ] ]. }
  destruct (existsb (fun '(x, kind) => is_dead_var sol x kind) bound)
    eqn:Hdead.
  - split; [ | split; intros k [] ].
    eexists. split; [ apply Hframe; intros H; exact H | constructor ].
  - destruct (IH _ _
                [(ret, (rds, true)); (exn, ([Pdesc de true Value], false))]
                (Some id) [(exn, [cd_exn cd']); (ret, cd_return cd')] Hci)
      as (IHuc & IHfc & IHbs & IHincl).
    { cbn [env_kinds bind_variables bind_variables_then bind_continuations
           empty_env].
      rewrite app_nil_r. exact Hcalls. }
    { split.
      - intros z Hz. apply Hvars. right. apply in_or_app. right. exact Hz.
      - intros k Hk. apply Hconts. right. right. exact Hk. }
    { exact Hbody_edges. }
    split; [ | split ].
    + eexists. split; [ apply Hframe; exact IHbs | ].
      apply code_K_of_body; [ cbn [code_body]; exact IHuc | ].
      exact (never_ret_dead id ret body _ Hnrb IHincl).
    + cbn [free_continuations_of_code]. apply incl_without_mono. exact IHfc.
    + intros k Hk.
      cbn [free_continuations_of_code live_conts_code] in Hk |- *.
      apply in_without_iff in Hk as [Hk Hne].
      apply in_without_iff. split; [ exact (IHincl k Hk) | exact Hne ].
Qed.

(* Code of the unit, as the two rebuilds make it. *)
Lemma code_rebuilt (id : code_id) (c : code) :
  rebuild_ok_expr (code_body c) ->
  calls_code_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
    nr reach ovr rem id
    c
  = true ->
  (∀ z, z ∈ variables_in_code c -> var_ok z) ->
  (∀ k, k ∈ continuations_in_code c -> k <> ff ∧ k <> ffu) ->
  FlambdaReaperFlowProofs.code_edges (graph_of_facts F) noted deps cds
    kept_slots codes0 symbols True False id c ->
  let ccc :=
    rebuild_code_cc F sol cannot_change code_deps pdc unit_code_ids codes ff
      id c
  in
  let cu :=
    rebuild_code_u true F sol cannot_change code_deps pdc unit_code_ids
      codes unboxed ffu [] id c
  in
  (∃ cs : code_shape,
     code_frame unboxed dead CSf id ccc cu cs
     ∧ uc unboxed dead CSf NRf RCHf OVf REMf (code_K_of NRf id ccc cs)
         (code_body ccc) (code_body cu))
  ∧ incl (free_continuations_of_code cu) (free_continuations_of_code ccc)
  ∧ incl (free_continuations_of_code cu)
      (live_conts_code F sol code_deps codes c).
Proof.
  intros IH Hok Hvars Hconts Hedges.
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed id
              (metadata_of_code c))
    eqn:Hch0;
    [ exact (code_rebuilt_nc id c IH Hok Hvars Hconts Hedges Hch0) | ].
  revert IH Hok Hvars Hconts Hedges Hch0.
  destruct c as [params mc ret exn md body].
  intros IH Hok Hvars Hconts Hedges Hch0. cbv zeta.
  cbn [metadata_of_code] in Hch0.
  cbn [calls_code_f3] in Hok. apply andb_true_iff in Hok as [Hok Hnrb].
  destruct (lookup id codes) as [md' | ] eqn:Hcodes;
    [ | discriminate Hok ].
  apply andb_true_iff in Hok as [Hmd Hok].
  apply code_metadata_eqb_eq in Hmd. subst md'.
  destruct (code_pdescs sol cannot_change code_deps unit_code_ids unboxed id
              md)
    as [[pds rds] | ] eqn:Hpd; [ | discriminate Hok ].
  unfold code_pdescs in Hpd.
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed id
              md)
    as [ | pd rd] eqn:Hchu; [ congruence | ].
  destruct (code_calling_convention_change sol cannot_change code_deps
              unit_code_ids id md)
    as [ | pd_cc rd_cc] eqn:Hchc; [ discriminate Hpd | ].
  injection Hpd as <- <-.
  pose proof Hchu as Hchu'. pose proof Hchc as Hchc'.
  apply andb_true_iff in Hok as [Hok Hcalls].
  destruct (code_exn_decision sol code_deps unboxed id) as [de | ] eqn:Hde;
    [ | discriminate Hcalls ].
  destruct (code_exn_decision_cases id de Hde) as (Hdc & cd0 & Hcd0 & Hfad).
  apply andb_true_iff in Hok as [Hok Hre].
  apply andb_true_iff in Hok as [Hok Hmc].
  apply andb_true_iff in Hok as [Hok Hmcd].
  apply andb_true_iff in Hok as [Hok Hpok].
  apply andb_true_iff in Hok as [Hok Hrok].
  apply andb_true_iff in Hok as [Hok Hdok].
  apply andb_true_iff in Hok as [Hok Hrlen].
  apply andb_true_iff in Hok as [Hlen Hdlen].
  apply Nat.eqb_eq in Hlen, Hdlen, Hrlen.
  apply negb_true_iff in Hre. apply Pos.eqb_neq in Hre.
  (* The decisions. *)
  unfold code_change_u in Hchu. unfold code_calling_convention_change in Hchc.
  destruct (cannot_change_calling_convention cannot_change unit_code_ids id);
    [ discriminate Hchu | ].
  destruct (lookup id code_deps) as [cd | ] eqn:Hcd; [ | discriminate Hchu ].
  injection Hchu as Hpd Hrd. injection Hchc as Hpd_cc Hrd_cc.
  assert (Hlink : Forall2 code_link pd pd_cc)
    by (rewrite <- Hpd, <- Hpd_cc; apply decisions_link).
  assert (Hrlink : Forall2 code_link rd rd_cc)
    by (rewrite <- Hrd, <- Hrd_cc; apply decisions_link).
  assert (Hlen_pd : length pd = length (unarize (cm_params_arity md))
                    ∧ length pd_cc = length (unarize (cm_params_arity md))).
  { rewrite pdescs_of_length, length_map in Hdlen.
    rewrite <- Hpd, <- Hpd_cc, decisions_u_length, decisions_of_length'.
    rewrite <- (Forall2_length Hlink) in Hdlen.
    rewrite <- Hpd, decisions_u_length in Hdlen.
    rewrite length_combine in Hdlen |- *. lia. }
  assert (Hlen_rd : length rd = length (unarize (cm_result_arity md))
                    ∧ length rd_cc = length (unarize (cm_result_arity md))).
  { rewrite pdescs_of_length, length_map in Hrlen.
    rewrite <- Hrd, <- Hrd_cc, decisions_u_length, decisions_of_length'.
    rewrite <- (Forall2_length Hrlink) in Hrlen.
    rewrite <- Hrd, decisions_u_length in Hrlen.
    rewrite length_combine in Hrlen |- *. lia. }
  destruct Hlen_pd as [Hlpd Hlpd_cc]. destruct Hlen_rd as [Hlrd Hlrd_cc].
  destruct (pdescs_of_decisions rd (map is_kept rd_cc)
              (unarize (cm_result_arity md)) Hlrd
              ltac:(rewrite length_map; exact Hlrd_cc))
    as [Hrdec Hrkeep].
  (* The edges of the body. *)
  destruct Hedges as (cd' & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _
                      & Hbody_edges).
  (* The rebuilt code. *)
  rewrite cc_rebuild_code, u_rebuild_code. cbv zeta.
  rewrite Hchc', Hchu'.
  destruct (my_closure_decision_u cannot_change code_deps unit_code_ids
              unboxed id);
    [ discriminate Hmcd | ].
  assert (Hccd : code_continuation_decisions sol code_deps unboxed id md
                   (Changing_u pd rd) ret exn
                 = [(ret, rd);
                    (exn, fixed_arity_decisions sol unboxed [cd_exn cd]
                            [Value])])
    by (unfold code_continuation_decisions; rewrite Hcd; reflexivity).
  rewrite Hccd.
  set (bound := (mc, Value)
                :: with_kinds params (unarize (cm_params_arity md))).
  set (rds := pdescs_of rd (map is_kept rd_cc) (unarize (cm_result_arity md)))
    in *.
  set (pds := pdescs_of pd (map is_kept pd_cc) (unarize (cm_params_arity md)))
    in *.
  assert (Hbound_ok : ∀ y kind, (y, kind) ∈ bound -> kind_ok kind = true).
  { intros y kind [Heq | Hin]; [ injection Heq as _ <-; reflexivity | ].
    apply (with_kinds_kinds_ok params (unarize (cm_params_arity md)) y);
      [ | exact Hin ].
    apply (pdescs_kinds_ok pd (map is_kept pd_cc)); [ | exact Hdok ].
    exact Hdlen. }
  assert (Hcd0' : cd0 = cd) by congruence. subst cd0.
  assert (Hci : cinv [(ret, (rds, true)); (exn, ([Pdesc de true Value], false))]
                  (bind_variables
                     (bind_continuations empty_env [(ret, map is_kept rd_cc)])
                     bound)
                  (ubind_variables
                     (ubind_continuations empty_unbox_env
                        [(ret, rd);
                         (exn, fixed_arity_decisions sol unboxed [cd_exn cd]
                                 [Value])])
                     bound)).
  { split.
    - intros k ps h Hl. rewrite !lookup_cons in Hl.
      destruct (Pos.eqb k ret); [ injection Hl as <- <-; exact Hrok | ].
      destruct (Pos.eqb k exn); [ | discriminate Hl ].
      injection Hl as <- <-. destruct Hdc as [-> | ->]; reflexivity.
    - reflexivity.
    - intros x. apply kind_of_bind; [ intros y; reflexivity | ].
      exact Hbound_ok.
    - reflexivity.
    - reflexivity.
    - reflexivity.
    - intros k. cbn [ue_decisions ubind_variables ubind_variables_then
                     ubind_set ubind_continuations empty_unbox_env app].
      rewrite !lookup_cons. destruct (Pos.eqb k ret) eqn:Hkr;
        [ cbn; rewrite Hrdec; reflexivity | ].
      destruct (Pos.eqb k exn) eqn:Hke; [ | reflexivity ].
      rewrite Hfad. reflexivity.
    - intros k. cbn [env_conts bind_variables bind_variables_then
                     bind_continuations empty_env app].
      rewrite !lookup_cons. destruct (Pos.eqb k ret);
        [ left; cbn; rewrite Hrkeep; reflexivity | ].
      destruct (Pos.eqb k exn) eqn:Hke; [ | left; reflexivity ].
      right. split; [ reflexivity | ].
      exists [Pdesc de true Value], false. split; reflexivity. }
  assert (Hcs : CSf id
                = Some
                    (Code_shape (flat_map pshape_of pds)
                       (flat_map pshape_of rds)
                       (flat_map kept_kind rds)
                       (arity_of_grouped_decisions_u
                          (prepend_first [Decided Delete]
                             (group_by_parameter (cm_params_arity md) pd)))
                       (arity_of_decisions_u rd)
                       (pshape1 (Pdesc de true Value)) false)).
  { unfold CSf. rewrite Hcodes, Hde, Hchu'. unfold code_pdescs.
    rewrite Hchu', Hchc'. reflexivity. }
  destruct (code_params_rel2 params pd pd_cc (unarize (cm_params_arity md))
              Hlink Hlen Hdok Hpok)
    as (sh & pk & pk' & Hpk & Hpk' & Hrel & Hfits).
  { intros x Hx. apply Hvars. right. apply in_or_app. left. exact Hx. }
  assert (Hmcf : fields_of unboxed mc = None).
  { cbn in Hmc. apply negb_true_iff in Hmc. unfold Keyed.is_bound in Hmc.
    unfold fields_of. destruct (Keyed.lookup _ _ _);
      [ discriminate Hmc | reflexivity ]. }
  destruct (existsb (fun '(x, kind) => is_dead_var sol x kind) bound)
    eqn:Hdead.
  - (* Both bodies are [Invalid]. *)
    split; [ | split; intros k [] ].
    + eexists. split.
      * split; [ exact Hcs | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split.
        { cbn [code_params_arity changed_code_metadata cm_params_arity
               cs_params_arity].
          unfold arity_of_grouped_decisions, arity_of_grouped_decisions_u.
          rewrite !length_map, prepend_first_length,
            !group_by_parameter_length.
          reflexivity. }
        split; [ intros H; discriminate H | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | split; [ exact Hmcf | ] ].
        split; [ exact (proj1 (Hvars mc (or_introl eq_refl))) | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ intros H; exact H | ].
        split; [ reflexivity | ].
        exists sh, pk, pk'. split; [ exact Hpk | ].
        split; [ cbn [code_params decision_params app];
                 rewrite flat_map_decision_params; exact Hpk' | ].
        split; [ exact Hrel | exact Hfits ].
      * constructor.
  - destruct (IH _ _
                [(ret, (rds, true)); (exn, ([Pdesc de true Value], false))]
                (Some id) [(exn, [cd_exn cd']); (ret, cd_return cd')] Hci)
      as (IHuc & IHfc & IHbs & IHincl).
    { cbn [env_kinds bind_variables bind_variables_then bind_continuations
           empty_env].
      rewrite app_nil_r. exact Hcalls. }
    { split.
      - intros z Hz. apply Hvars. right. apply in_or_app. right. exact Hz.
      - intros k Hk. apply Hconts. right. right. exact Hk. }
    { exact Hbody_edges. }
    split; [ | split ].
    + eexists. split.
      * split; [ exact Hcs | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split.
        { cbn [code_params_arity changed_code_metadata cm_params_arity
               cs_params_arity].
          unfold arity_of_grouped_decisions, arity_of_grouped_decisions_u.
          rewrite !length_map, prepend_first_length,
            !group_by_parameter_length.
          reflexivity. }
        split; [ intros H; discriminate H | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | split; [ exact Hmcf | ] ].
        split; [ exact (proj1 (Hvars mc (or_introl eq_refl))) | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ cbn [code_body]; exact IHbs | ].
        split; [ reflexivity | ].
        exists sh, pk, pk'. split; [ exact Hpk | ].
        split; [ cbn [code_params decision_params app];
                 rewrite flat_map_decision_params; exact Hpk' | ].
        split; [ exact Hrel | exact Hfits ].
      * apply code_K_of_body; [ cbn [code_body]; exact IHuc | ].
        exact (never_ret_dead id ret body _ Hnrb IHincl).
    + cbn [free_continuations_of_code]. apply incl_without_mono. exact IHfc.
    + intros k Hk.
      cbn [free_continuations_of_code live_conts_code] in Hk |- *.
      apply in_without_iff in Hk as [Hk Hne].
      apply in_without_iff. split; [ exact (IHincl k Hk) | exact Hne ].
Qed.

Lemma cc_rebuild_binding_code (en : rebuild_env) (id : code_id) (c : code) :
  rebuild_binding_cc F sol cannot_change code_deps pdc unit_code_ids codes ff
    en (Bind_code id c)
  = if is_code_id_used sol id
    then [Bind_code id (rebuild_code_cc F sol cannot_change code_deps pdc
                          unit_code_ids codes ff id c)]
    else [Bind_deleted_code id].
Proof. reflexivity. Qed.

Lemma u_rebuild_binding_code (enu : unbox_env) (id : code_id) (c : code) :
  rebuild_binding_u true F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu [] enu (Bind_code id c)
  = if is_code_id_used sol id
    then [Bind_code id (rebuild_code_u true F sol cannot_change code_deps
                          pdc unit_code_ids codes unboxed ffu [] id c)]
    else [Bind_deleted_code id].
Proof. reflexivity. Qed.

Lemma cc_rebuild_binding_deleted (en : rebuild_env) (id : code_id) :
  rebuild_binding_cc F sol cannot_change code_deps pdc unit_code_ids codes ff
    en (Bind_deleted_code id)
  = [Bind_deleted_code id].
Proof. reflexivity. Qed.

Lemma u_rebuild_binding_deleted (enu : unbox_env) (id : code_id) :
  rebuild_binding_u true F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu [] enu (Bind_deleted_code id)
  = [Bind_deleted_code id].
Proof. reflexivity. Qed.

Lemma cc_rebuild_binding_block (en : rebuild_env) (sym : symbol) (tag : Z)
    (fields : list simple) :
  rebuild_binding_cc F sol cannot_change code_deps pdc unit_code_ids codes ff
    en (Bind_symbol sym (Static_const_block tag fields))
  = if is_symbol_used sol sym
    then [Bind_symbol sym
            (Static_const_block tag
               (rebuild_fields sol en (Node_symbol sym) 0 fields))]
    else [].
Proof. reflexivity. Qed.

Lemma u_rebuild_binding_block (enu : unbox_env) (sym : symbol) (tag : Z)
    (fields : list simple) :
  rebuild_binding_u true F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu [] enu (Bind_symbol sym (Static_const_block tag fields))
  = if is_symbol_used sol sym
    then [Bind_symbol sym
            (Static_const_block tag
               (rebuild_fields sol_u (ue_env enu) (Node_symbol sym) 0
                  fields))]
    else [].
Proof. reflexivity. Qed.

Lemma rebuild_fields_plain2 (en : rebuild_env) (base : node)
    (args : list simple) :
  (∀ y, kind_ok (kind_of_variable en y) = true) ->
  forallb (plain_simple unboxed) args = true ->
  ∀ i, forallb (plain_simple2 unboxed dead)
         (rebuild_fields sol en base i args) = true.
Proof.
  intros Hk. induction args as [ | a args IH]; intros Ha i;
    [ reflexivity | ].
  simpl in Ha |- *. apply andb_true_iff in Ha as [Ha Hargs].
  rewrite (IH Hargs), andb_true_r.
  destruct (field_used sol _ _); [ | reflexivity ].
  apply rewrite_plain2; assumption.
Qed.

(* Static bindings, code included, as the two rebuilds make them. *)
Lemma bindings_rebuilt (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (bindings : list static_binding) :
  (∀ (id : code_id) (c : code),
     Bind_code id c ∈ bindings -> rebuild_ok_expr (code_body c)) ->
  cinv C en enu ->
  forallb
    (calls_binding_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach ovr rem)
    bindings
  = true ->
  (∀ z, z ∈ flat_map variables_in_binding bindings -> var_ok z) ->
  (∀ k, k ∈ flat_map continuations_in_binding bindings -> k <> ff ∧ k <> ffu) ->
  every
    (FlambdaReaperFlowProofs.binding_edges (graph_of_facts F) noted deps cds
       kept_slots codes0 symbols True False)
    bindings ->
  let bcc :=
    flat_map
      (rebuild_binding_cc F sol cannot_change code_deps pdc unit_code_ids
         codes ff en)
      bindings
  in
  let bu :=
    flat_map
      (rebuild_binding_u true F sol cannot_change code_deps pdc
         unit_code_ids codes unboxed ffu [] enu)
      bindings
  in
  uc_bindings unboxed dead CSf NRf RCHf OVf REMf bcc bu
  ∧ incl (flat_map free_continuations_of_binding bu)
      (flat_map free_continuations_of_binding bcc)
  ∧ existsb binding_binds bcc = existsb binding_binds bu
  ∧ incl (flat_map free_continuations_of_binding bu)
      (flat_map (live_conts_binding F sol code_deps codes) bindings).
Proof.
  intros IHcode Hci Hok Hvars Hconts Hedges. cbv zeta.
  induction bindings as [ | b bs IHbs].
  - split; [ constructor | ].
    split; [ intros k [] | split; [ reflexivity | intros k [] ] ].
  - cbn [forallb] in Hok. apply andb_true_iff in Hok as [Hb Hbs].
    destruct Hedges as [Hb_edges Hbs_edges].
    destruct (IHbs (fun id c Hin => IHcode id c (or_intror Hin)) Hbs)
      as (Hub & Hfc & Hbb & Hincl).
    { intros z Hz. apply Hvars. cbn [flat_map]. apply in_or_app. right.
      exact Hz. }
    { intros k Hk. apply Hconts. cbn [flat_map]. apply in_or_app. right.
      exact Hk. }
    { exact Hbs_edges. }
    destruct b as [id c | id | sym const]; cbn [flat_map].
    + cbn [calls_binding_f3] in Hb.
      rewrite cc_rebuild_binding_code, u_rebuild_binding_code.
      destruct (is_code_id_used sol id).
      * destruct Hb_edges as [_ Hcode_edges].
        destruct (code_rebuilt id c (IHcode id c (or_introl eq_refl)) Hb)
          as ((cs & Hframe & Hcuc) & Hcfc & Hcincl).
        { intros z Hz. apply Hvars. cbn [flat_map variables_in_binding].
          apply in_or_app. left. exact Hz. }
        { intros k Hk. apply Hconts. cbn [flat_map continuations_in_binding].
          apply in_or_app. left. exact Hk. }
        { exact Hcode_edges. }
        cbn [app].
        split;
          [ apply (uc_bindings_code _ _ _ _ _ _ _ _ _ _ cs); assumption | ].
        cbn [flat_map free_continuations_of_binding existsb binding_binds
             orb app live_conts_binding].
        split; [ apply incl_app_app; [ exact Hcfc | exact Hfc ] | ].
        split; [ reflexivity | ].
        intros k Hk. apply in_app_or in Hk as [Hk | Hk]; apply in_or_app;
          [ left; exact (Hcincl k Hk) | right; exact (Hincl k Hk) ].
      * cbn [app].
        split; [ apply uc_bindings_deleted; exact Hub | ].
        cbn [flat_map free_continuations_of_binding existsb binding_binds
             orb app live_conts_binding].
        split; [ exact Hfc | split; [ exact Hbb | ] ].
        intros k Hk. apply in_or_app. right. exact (Hincl k Hk).
    + rewrite cc_rebuild_binding_deleted, u_rebuild_binding_deleted.
      cbn [app].
      split; [ apply uc_bindings_deleted; exact Hub | ].
      cbn [flat_map free_continuations_of_binding existsb binding_binds orb
           app live_conts_binding].
      split; [ exact Hfc | split; [ exact Hbb | ] ].
      intros k Hk. exact (Hincl k Hk).
    + cbn [calls_binding_f3] in Hb.
      destruct const as [ | | tag fields | | | ]; cbn [local_binding] in Hb;
        try discriminate Hb.
      assert (Hfields : forallb (plain_simple unboxed) fields = true).
      { apply (strong_simples2 fields Hb). intros z Hz.
        exact (proj1 (Hvars z (in_or_app _ _ _ (or_introl Hz)))). }
      rewrite cc_rebuild_binding_block, u_rebuild_binding_block.
      destruct (is_symbol_used sol sym).
      * cbn [app].
        rewrite (rebuild_fields_kinds sol_u (ue_env enu) en _ fields
                   (ci_kinds _ _ _ Hci) 0),
          (rebuild_fields_u sol unboxed en _ fields Hfields 0).
        split; [ apply uc_bindings_symbol; [ | exact Hub ] | ].
        { cbn [local_binding2]. apply rebuild_fields_plain2;
            [ exact (ci_kinds_ok _ _ _ Hci) | exact Hfields ]. }
        cbn [flat_map free_continuations_of_binding existsb binding_binds
             orb app live_conts_binding].
        split; [ exact Hfc | split; [ reflexivity | ] ].
        intros k Hk. exact (Hincl k Hk).
      * cbn [app live_conts_binding]. split; [ exact Hub | ].
        split; [ exact Hfc | split; [ exact Hbb | ] ].
        intros k Hk. exact (Hincl k Hk).
Qed.


Lemma args_are_params_eq (args : list simple) :
  ∀ ps : list variable, args_are_params args ps = true -> args = map Var ps.
Proof.
  induction args as [ | a args IH]; intros [ | p ps] H;
    [ reflexivity | discriminate H | destruct a; discriminate H | ].
  destruct a as [x | | ]; simpl in H; try discriminate H.
  apply andb_true_iff in H as [Hx H]. apply Pos.eqb_eq in Hx. subst x.
  rewrite (IH ps H). reflexivity.
Qed.

Lemma eval_bound_vars (ps : list variable) :
  ∀ (vs : list value) (wv hv : env),
    NoDup ps -> bind_params ps vs wv = Some hv ->
    eval_simples hv (map Var ps) = Some vs.
Proof.
  intros vs wv hv Hnd Hb.
  assert (Hlen : length ps = length vs)
    by exact (bind_params_length _ _ _ _ Hb).
  assert (H : ∀ i x, nth_error ps i = Some x ->
              ∃ v, nth_error vs i = Some v ∧ lookup x hv = Some v).
  { intros i x Hx.
    destruct (nth_error vs i) as [v | ] eqn:Hv.
    - exists v. split; [ reflexivity | ].
      exact (bind_params_nth ps vs wv hv i x v Hnd Hb Hx Hv).
    - exfalso. apply nth_error_None in Hv.
      assert (Hlt : i < length ps)
        by (apply nth_error_Some; rewrite Hx; discriminate). lia. }
  clear Hb Hnd. revert vs Hlen H.
  induction ps as [ | x ps IH]; intros [ | v vs] Hlen H; simpl in Hlen;
    try discriminate Hlen; [ reflexivity | ].
  destruct (H 0 x eq_refl) as (v0 & Hv0 & Hl). simpl in Hv0.
  injection Hv0 as <-.
  cbn [map eval_simples eval_simple]. rewrite Hl.
  rewrite (IH vs ltac:(lia) (fun i y Hy => H (S i) y Hy)). reflexivity.
Qed.

Lemma call_rel_with (ap ap' : apply_expr) (cs : code_shape)
    (w w' : continuation) :
  call_rel unboxed dead CSf RCHf OVf REMf ap ap' cs ->
  call_rel unboxed dead CSf RCHf OVf REMf (with_continuation ap w)
      (with_continuation ap'
    w')
    cs.
Proof. intros H. exact H. Qed.

Lemma uc_result_call (K : finite_map kshape) (ap0 ap ap' : apply_expr)
    (cs : code_shape) (k : continuation) (sh : list pshape) :
  call_rel unboxed dead CSf RCHf OVf REMf ap ap' cs ->
  exn_call_rel unboxed dead K (cs_exn cs) ap ap' ->
  ap_continuation ap = k -> ap_continuation ap' = k ->
  ap_exn_continuation ap = ap_exn_continuation ap0 ->
  ap_continuation ap0 = k ->
  lookup k K = Some (Kshape sh (cs_result_kinds cs) true) ->
  transport (cs_results cs) sh (cs_result_kinds cs) ->
  uc_result K (free_continuations_of_expr (Apply ap0)) (Apply ap) (Apply ap').
Proof.
  intros Hcall Hexn Hk Hk' He Hk0 Hlk Ht.
  split; [ exact (uc_call unboxed dead CSf NRf RCHf OVf REMf K ap ap' cs k
                    sh Hcall Hexn Hk Hk' Hlk Ht) | ].
  pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He').
  cbn [free_continuations_of_expr binds_statics].
  rewrite Hk, Hk', He'.
  split; [ intros x Hx; exact Hx | split; [ intros H; exact H | ] ].
  rewrite He, Hk0. intros x Hx. exact Hx.
Qed.

Lemma fc_wrap (w : continuation) (ps : list variable)
    (md : handler_metadata) (k : continuation) (args : list simple)
    (ap : apply_expr) :
  ap_continuation ap = w -> ap_exn_continuation ap <> w ->
  free_continuations_of_expr (wrap w ps md k args (Apply ap))
  = [k; ap_exn_continuation ap].
Proof.
  intros Hw He. unfold wrap. cbn [free_continuations_of_expr
    free_continuations_of_handler apply_cont_continuations ac_continuation
    ac_trap_action app]. rewrite Hw. unfold without. cbn [filter].
  rewrite !among_cons, Pos.eqb_refl. cbn [orb negb].
  rewrite (proj2 (Pos.eqb_neq _ _) He). reflexivity.
Qed.

Lemma uc_result_call_wrap (K : finite_map kshape) (ap0 ap ap' : apply_expr)
    (cs : code_shape) (k : continuation) (sh : list pshape)
    (w : continuation) (wps : list variable) (wmd : handler_metadata)
    (wargs : list simple) :
  call_rel unboxed dead CSf RCHf OVf REMf ap ap' cs ->
  exn_call_rel unboxed dead K (cs_exn cs) ap ap' ->
  ap_continuation ap = k -> ap_continuation ap' = w ->
  w <> ap_exn_continuation ap ->
  ap_exn_continuation ap = ap_exn_continuation ap0 ->
  ap_continuation ap0 = k ->
  lookup k K = Some (Kshape sh (cs_result_kinds cs) true) ->
  wrap_u_ok (cs_results cs) sh (cs_result_kinds cs) wps wmd wargs ->
  uc_result K (free_continuations_of_expr (Apply ap0)) (Apply ap)
    (wrap w wps wmd k wargs (Apply ap')).
Proof.
  intros Hcall Hexn Hk Hw Hwe He Hk0 Hlk Hok.
  split; [ exact (uc_call_wrap unboxed dead CSf NRf RCHf OVf REMf K ap ap'
                    cs k sh w wps wmd wargs Hcall Hexn Hk Hw Hwe Hlk Hok) | ].
  pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He').
  rewrite fc_wrap by (try exact Hw; rewrite He'; intros H; apply Hwe;
                      symmetry; exact H).
  cbn [free_continuations_of_expr binds_statics].
  rewrite Hk, He'.
  split; [ intros x Hx; exact Hx | split; [ intros H; exact H | ] ].
  rewrite He, Hk0. intros x Hx. exact Hx.
Qed.

Lemma uc_result_call_wraps (K : finite_map kshape) (ap0 ap ap' : apply_expr)
    (cs : code_shape) (k : continuation) (sh : list pshape)
    (KS : list flambda_kind) (wc : continuation) (cps : list variable)
    (cmd : handler_metadata) (cargs : list simple) (w : continuation)
    (wps : list variable) (wmd : handler_metadata) (wargs : list simple) :
  call_rel unboxed dead CSf RCHf OVf REMf ap ap' cs ->
  exn_call_rel unboxed dead K (cs_exn cs) ap ap' ->
  ap_continuation ap = wc -> ap_continuation ap' = w ->
  wc <> ap_exn_continuation ap -> w <> ap_exn_continuation ap ->
  ap_exn_continuation ap = ap_exn_continuation ap0 ->
  ap_continuation ap0 = k ->
  lookup k K = Some (Kshape sh KS true) ->
  hm_params_kinds cmd = cs_result_kinds cs ->
  wrap_both_ok (cs_results cs) sh cps cmd cargs wps wmd wargs ->
  uc_result K (free_continuations_of_expr (Apply ap0))
    (wrap wc cps cmd k cargs (Apply ap)) (wrap w wps wmd k wargs (Apply ap')).
Proof.
  intros Hcall Hexn Hwc Hw Hwce Hwe He Hk0 Hlk Hcmd Hok.
  split; [ exact (uc_call_wraps unboxed dead CSf NRf RCHf OVf REMf K ap ap'
                    cs k sh KS wc cps cmd cargs w wps wmd wargs Hcall Hexn
                    Hwc Hw Hwce Hwe Hlk Hcmd Hok) | ].
  pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He').
  rewrite (fc_wrap wc) by (try exact Hwc; intros H; apply Hwce;
                           symmetry; exact H).
  rewrite (fc_wrap w) by (try exact Hw; rewrite He'; intros H; apply Hwe;
                          symmetry; exact H).
  rewrite He'.
  split; [ intros x Hx; exact Hx | split; [ intros H; exact H | ] ].
  rewrite He. cbn [free_continuations_of_expr]. rewrite Hk0.
  intros x Hx. exact Hx.
Qed.

(* The call of the first run, as [make_apply_wrapper] makes it: the call,
   or the call returning to a wrapper. *)
Definition call_or_wrap (p : expr) (ap : apply_expr) : Prop :=
  p = Apply ap
  ∨ ∃ (cps : list variable) (cmd : handler_metadata) (cargs : list simple),
      p = wrap ff cps cmd (ap_continuation ap) cargs
            (Apply (with_continuation ap ff)).

Lemma make_apply_wrapper_cases (en : rebuild_env) (ap : apply_expr) :
  call_or_wrap (make_apply_wrapper ff en ap) ap.
Proof.
  unfold call_or_wrap, make_apply_wrapper.
  destruct (lookup _ (env_conts en)) as [mask | ]; [ | left; reflexivity ].
  destruct (forallb _ mask); [ left; reflexivity | right ].
  do 3 eexists. reflexivity.
Qed.

Lemma make_apply_wrapper_cc_cases (en : rebuild_env) (ap : apply_expr)
    (fd : list (param_decision * flambda_kind)) :
  call_or_wrap (make_apply_wrapper_cc ff en ap fd) ap.
Proof.
  unfold call_or_wrap, make_apply_wrapper_cc. cbv zeta.
  destruct (forallb _ _); [ left; reflexivity | right ].
  do 3 eexists. reflexivity.
Qed.

(* A call of code taken to never return, which [apply_never_returns]: the
   second run gives it a handler [Invalid]. *)
Lemma uc_result_never (K : finite_map kshape) (fc0 : list continuation)
    (ap ap' : apply_expr) (cs : code_shape) (id : code_id) (p : expr)
    (enu : unbox_env) (fd : list (decision * flambda_kind)) :
  call_rel unboxed dead CSf RCHf OVf REMf ap ap' cs ->
  exn_call_rel unboxed dead K (cs_exn cs) ap ap' ->
  ap_call_kind ap = Direct id -> CSf id = Some cs -> NRf id = true ->
  ff <> ap_exn_continuation ap -> ffu <> ap_exn_continuation ap ->
  call_or_wrap p ap ->
  incl [ap_exn_continuation ap] fc0 ->
  uc_result K fc0 p (make_apply_wrapper_u true ffu enu ap' fd true).
Proof.
  intros Hcall Hexn Hkind Hcs Hnr Hffe Hffue Hp Hfc0.
  pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & He').
  unfold make_apply_wrapper_u. cbv zeta. cbn [andb].
  set (wps := map fst (wrapper_params (freshen_decisions fd ffu))).
  set (wmd := make_handler_metadata
                (map snd (wrapper_params (freshen_decisions fd ffu)))).
  assert (Hfcq : free_continuations_of_expr
                   (Let_cont_non_recursive ffu (Handler wps wmd Invalid)
                      (Apply (with_continuation ap' ffu)))
                 = [ap_exn_continuation ap]).
  { cbn [free_continuations_of_expr free_continuations_of_handler app
         ap_continuation ap_exn_continuation with_continuation].
    rewrite He'. unfold without. cbn [filter].
    rewrite !among_cons, Pos.eqb_refl. cbn [orb negb].
    rewrite (proj2 (Pos.eqb_neq _ _) (fun H => Hffue (eq_sym H))).
    reflexivity. }
  assert (Hin : ∀ q,
             ap_exn_continuation ap ∈ free_continuations_of_expr q ->
             incl [ap_exn_continuation ap] (free_continuations_of_expr q)).
  { intros q Hq x [<- | []]. exact Hq. }
  split; [ | split; [ | split ] ].
  - destruct Hp as [-> | (cps & cmd & cargs & ->)].
    + exact (uc_call_never unboxed dead CSf NRf RCHf OVf REMf K ap
               (with_continuation ap' ffu) cs id ffu wps wmd Hcall Hexn Hkind
               Hcs Hnr eq_refl Hffue).
    + exact (uc_call_never_wraps unboxed dead CSf NRf RCHf OVf REMf K
               (with_continuation ap ff) (with_continuation ap' ffu) cs id ff
               cps cmd (ap_continuation ap) cargs ffu wps wmd Hcall Hexn Hkind
               Hcs Hnr eq_refl eq_refl Hffe Hffue).
  - rewrite Hfcq. apply Hin.
    destruct Hp as [-> | (cps & cmd & cargs & ->)];
      [ right; left; reflexivity | ].
    rewrite fc_wrap;
      [ right; left; reflexivity | reflexivity
      | intros H; exact (Hffe (eq_sym H)) ].
  - cbn [binds_statics handler_binds_statics orb]. intros H. discriminate H.
  - rewrite Hfcq. exact Hfc0.
Qed.

Lemma singletons_snd {A : Type} (l : list (A * flambda_kind)) :
  map (fun '(_, kind) => Singleton kind) l = map Singleton (map snd l).
Proof. induction l as [ | [a k] l IH]; simpl; [ | rewrite IH ]; reflexivity.
Qed.

Lemma forallb2_eqb_eq (l : list bool) :
  ∀ l' : list bool, forallb2 Bool.eqb l l' = true -> l = l'.
Proof.
  induction l as [ | b l IH]; intros [ | b' l'] H; simpl in H;
    try discriminate H; [ reflexivity | ].
  apply andb_true_iff in H as [Hb H]. apply Bool.eqb_prop in Hb. subst b'.
  rewrite (IH l' H). reflexivity.
Qed.

Lemma mask_arguments_kinds (en1 en2 : rebuild_env) (args : list simple) :
  env_kinds en1 = env_kinds en2 ->
  ∀ mask : list bool,
    mask_arguments en1 mask args = mask_arguments en2 mask args.
Proof.
  intros Hk. induction args as [ | a args IH]; intros [ | b mask];
    simpl; try reflexivity; rewrite ?IH; [ reflexivity | ].
  destruct b; [ reflexivity | ]. destruct a; simpl; [ | reflexivity
                                                      | reflexivity ].
  unfold kind_of_variable. rewrite Hk. reflexivity.
Qed.

Lemma plain_shapes (kinds : list flambda_kind) :
  flat_map pshape_of
    (map (fun kind => Pdesc (Decided (Keep kind)) true kind) kinds)
  = map (fun _ => Psh_plain) kinds.
Proof.
  induction kinds as [ | k kinds IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma plain_shapes_kinds (kinds ks : list flambda_kind) :
  length ks = length kinds ->
  shape_kinds (map (fun _ => Psh_plain) kinds) ks = ks.
Proof.
  revert ks. induction kinds as [ | k kinds IH]; intros [ | k' ks] H;
    simpl in H |- *; try discriminate H; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma plain_shapes_no_dead (kinds : list flambda_kind) :
  no_dead (map (fun _ => Psh_plain) kinds).
Proof.
  intros kind Hin. apply in_map_iff in Hin as (k & Heq & _). discriminate Heq.
Qed.

(* The arguments of a call at a mask of both runs: those that the mask
   keeps as they are, the others poison, of the kind of the argument, an
   unboxed name among them. *)
Lemma args_rel2_mask (en : rebuild_env) (args : list simple) :
  ∀ (mask : list bool) (kinds : list flambda_kind),
    length args = length kinds ->
    forallb2
      (fun keep a =>
         negb keep
         || FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed a
            && match a with
               | Var z => s_has_usage sol (Node_variable z)
               | Symbol _ | Const _ => true
               end)
      mask args
    = true ->
    (∀ z, z ∈ variables_in_simples args -> fresh z) ->
    args_rel2 unboxed dead (map (fun _ => Psh_plain) kinds)
      (mask_arguments en mask args) (mask_arguments en mask args).
Proof.
  induction args as [ | a args IH]; intros [ | b mask] [ | k kinds] Hlen
    Hused Hfr; simpl in Hlen, Hused; try discriminate Hlen;
    try discriminate Hused.
  - apply args_rel2_beyond. reflexivity.
  - apply andb_true_iff in Hused as [Hb Hused].
    cbn [map mask_arguments]. apply args_rel2_plain;
      [ | refine (IH mask kinds ltac:(lia) Hused _);
          intros z Hz; apply Hfr; cbn [variables_in_simples];
          apply in_or_app; right; exact Hz ].
    destruct b; cbn [negb orb] in Hb.
    + apply andb_true_iff in Hb as [Ha Hb].
      assert (Ha' : plain_simple unboxed a = true).
      { apply strong_simple2; [ exact Ha | ].
        intros z Hz. apply Hfr. cbn [variables_in_simples].
        apply in_or_app. left. exact Hz. }
      destruct a as [z | sym | c]; [ | exact Ha' | reflexivity ].
      apply plain2_iff. split; [ exact Ha' | ].
      destruct (among z dead) eqn:Hd; [ | reflexivity ].
      rewrite (dead_unused z Hd) in Hb. discriminate Hb.
    + destruct a; reflexivity.
Qed.


Lemma args_ok2_keep (kenv : finite_map flambda_kind) (args : list simple) :
  ∀ kinds : list flambda_kind,
    length args = length kinds ->
    args_ok2 sol unboxed kenv
      (map (fun kind => Pdesc (Decided (Keep kind)) true kind) kinds) args
    = true ->
    forallb (FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed) args
    = true.
Proof.
  induction args as [ | a args IH]; intros [ | k kinds] Hlen H;
    simpl in Hlen; try discriminate Hlen; [ reflexivity | ].
  cbn [map args_ok2] in H. apply andb_true_iff in H as [Ha H].
  cbn [forallb]. rewrite (IH kinds ltac:(lia) H), andb_true_r. exact Ha.
Qed.

(* Values of the shape of one result that the callee keeps, used or not. *)
Lemma vrel_one_kept (sh : list pshape) (vs vs' : list value)
    (kind : flambda_kind) :
  sh = [Psh_plain] ∨ sh = [Psh_dead kind] ->
  vrel sh vs vs' -> map kind_of_value vs = [kind] ->
  ∃ v v' : value,
    vs = [v] ∧ vs' = [v'] ∧ kind_of_value v' = kind ∧ vrel sh [v] [v'].
Proof.
  intros [-> | ->] Hv Hk.
  - destruct (vrel_one_plain vs vs' kind Hv Hk) as (v & -> & ->).
    exists v, v. cbn in Hk. injection Hk as Hk.
    split; [ reflexivity | split; [ reflexivity | split; [ exact Hk | ] ] ].
    exact Hv.
  - inversion Hv; subst; [ discriminate Hk | ].
    match goal with
    | H : vrel [] ?a ?b |- _ => inversion H; subst
    end.
    match goal with
    | H : map kind_of_value (?v :: ?l) = [_] |- _ =>
      destruct l; [ | discriminate H ]; cbn in H; injection H as Hkv
    end.
    eexists _, _. split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ | exact Hv ].
    match goal with
    | H : _ ∨ _ |- _ => destruct H as [-> | H]; [ exact Hkv | exact H ]
    end.
Qed.

(* ======================================================================== *)
(* Wrappers, result by result                                               *)
(* ======================================================================== *)

(* The parameters [ps] hold the values [vs] in [hv]. *)
Definition binds (ps : list variable) (vs : list value) (hv : env) : Prop :=
  Forall2 (fun p v => lookup p hv = Some v) ps vs.

(* What a wrapper of each run makes of some results of a call: given what
   the callee returns, of the shapes [sh_r] (the first run's values of the
   kinds [KS]), the first run's wrapper, whose parameters are [cps], gives
   [cargs], the second's, whose parameters are [wps], gives [wargs], and
   the two are of the shapes [sh_k] of the continuation. *)
Definition seg_ok (sh_r sh_k : list pshape) (KS : list flambda_kind)
    (cps : list variable) (cargs : list simple)
    (wps : list (variable * flambda_kind)) (wargs : list simple) : Prop :=
  length sh_r = length KS ∧ length cps = length KS
  ∧ ∀ vs vs' : list value,
      vrel sh_r vs vs' -> map kind_of_value vs = KS ->
      map kind_of_value vs' = map snd wps
      ∧ ∀ hv hv' : env,
          binds cps vs hv -> binds (map fst wps) vs' hv' ->
          ∃ ws ws' : list value,
            eval_simples hv cargs = Some ws
            ∧ eval_simples hv' wargs = Some ws'
            ∧ length ws = length sh_k ∧ vrel sh_k ws ws'.

Lemma vrel_app (sh1 sh2 : list pshape) (vs1 vs1' vs2 vs2' : list value) :
  vrel sh1 vs1 vs1' -> length vs1 = length sh1 -> vrel sh2 vs2 vs2' ->
  vrel (sh1 ++ sh2) (vs1 ++ vs2) (vs1' ++ vs2').
Proof.
  induction 1 as [vs | o sh | sh v vs vs' _ IH | sh kind v v' vs vs' Hv _ IH
                 | sh fields v ws vs vs' Hws _ IH];
    intros Hlen H2.
  - destruct vs; [ exact H2 | discriminate Hlen ].
  - discriminate Hlen.
  - cbn [app]. constructor. apply IH; [ simpl in Hlen; lia | exact H2 ].
  - cbn [app]. constructor; [ exact Hv | ].
    apply IH; [ simpl in Hlen; lia | exact H2 ].
  - cbn [app]. rewrite <- app_assoc. constructor; [ exact Hws | ].
    apply IH; [ simpl in Hlen; lia | exact H2 ].
Qed.

Lemma vrel_app_inv (sh1 : list pshape) :
  ∀ (sh2 : list pshape) (vs1 vs2 vs' : list value),
    vrel (sh1 ++ sh2) (vs1 ++ vs2) vs' -> length vs1 = length sh1 ->
    ∃ a b : list value, vs' = a ++ b ∧ vrel sh1 vs1 a ∧ vrel sh2 vs2 b.
Proof.
  induction sh1 as [ | o sh1 IH]; intros sh2 vs1 vs2 vs' Hv Hlen.
  - destruct vs1; [ | discriminate Hlen ].
    exists [], vs'. split; [ reflexivity | split; [ | exact Hv ] ].
    constructor.
  - destruct vs1 as [ | v vs1]; [ discriminate Hlen | ].
    simpl in Hlen. injection Hlen as Hlen. cbn [app] in Hv.
    inversion Hv; subst.
    + destruct (IH sh2 vs1 vs2 vs'0 ltac:(eassumption) Hlen)
        as (a & b & -> & Ha & Hb).
      exists (v :: a), b. split; [ reflexivity | ].
      split; [ constructor; exact Ha | exact Hb ].
    + destruct (IH sh2 vs1 vs2 vs'0 ltac:(eassumption) Hlen)
        as (a & b & -> & Ha & Hb).
      exists (v' :: a), b. split; [ reflexivity | ].
      split; [ constructor; assumption | exact Hb ].
    + destruct (IH sh2 vs1 vs2 vs'0 ltac:(eassumption) Hlen)
        as (a & b & -> & Ha & Hb).
      exists (ws ++ a), b. split; [ apply app_assoc | ].
      split; [ constructor; assumption | exact Hb ].
Qed.

Lemma binds_app (ps1 ps2 : list variable) (vs1 vs2 : list value) (hv : env) :
  binds (ps1 ++ ps2) (vs1 ++ vs2) hv -> length ps1 = length vs1 ->
  binds ps1 vs1 hv ∧ binds ps2 vs2 hv.
Proof.
  revert vs1. induction ps1 as [ | p ps1 IH]; intros [ | v vs1] Hb Hlen;
    simpl in Hlen; try discriminate Hlen.
  - split; [ constructor | exact Hb ].
  - inversion Hb as [ | p0 v0 l l' Hp Hrest]; subst.
    destruct (IH vs1 Hrest ltac:(lia)) as [Ha Hb2].
    split; [ constructor; assumption | exact Hb2 ].
Qed.

Lemma binds_length (ps : list variable) (vs : list value) (hv : env) :
  binds ps vs hv -> length ps = length vs.
Proof. exact (@Forall2_length _ _ _ ps vs). Qed.

Lemma seg_nil : seg_ok [] [] [] [] [] [] [].
Proof.
  split; [ reflexivity | split; [ reflexivity | ] ].
  intros vs vs' Hv Hk. destruct vs; [ | discriminate Hk ].
  apply vrel_none in Hv. subst vs'.
  split; [ reflexivity | ].
  intros hv hv' _ _. exists [], [].
  split; [ reflexivity | split; [ reflexivity | ] ].
  split; [ reflexivity | constructor ].
Qed.

Lemma seg_app (shr1 shk1 : list pshape) (KS1 : list flambda_kind)
    (cps1 : list variable) (cargs1 : list simple)
    (wps1 : list (variable * flambda_kind)) (wargs1 : list simple)
    (shr2 shk2 : list pshape) (KS2 : list flambda_kind)
    (cps2 : list variable) (cargs2 : list simple)
    (wps2 : list (variable * flambda_kind)) (wargs2 : list simple) :
  seg_ok shr1 shk1 KS1 cps1 cargs1 wps1 wargs1 ->
  seg_ok shr2 shk2 KS2 cps2 cargs2 wps2 wargs2 ->
  seg_ok (shr1 ++ shr2) (shk1 ++ shk2) (KS1 ++ KS2) (cps1 ++ cps2)
    (cargs1 ++ cargs2) (wps1 ++ wps2) (wargs1 ++ wargs2).
Proof.
  intros (Hl1 & Hc1 & H1) (Hl2 & Hc2 & H2).
  split; [ rewrite !length_app; lia | split; [ rewrite !length_app; lia | ] ].
  intros vs vs' Hv Hk.
  apply map_eq_app in Hk as (a1 & a2 & -> & Hk1 & Hk2).
  assert (Ha1 : length a1 = length shr1)
    by (rewrite Hl1, <- Hk1, length_map; reflexivity).
  destruct (vrel_app_inv shr1 shr2 a1 a2 vs' Hv Ha1)
    as (b1 & b2 & -> & Hv1 & Hv2).
  destruct (H1 a1 b1 Hv1 Hk1) as (Hkb1 & H1').
  destruct (H2 a2 b2 Hv2 Hk2) as (Hkb2 & H2').
  split; [ rewrite !map_app, Hkb1, Hkb2; reflexivity | ].
  intros hv hv' Hb Hb'.
  apply binds_app in Hb as [Hb1 Hb2];
    [ | rewrite Hc1, <- Hk1, length_map; reflexivity ].
  rewrite map_app in Hb'.
  apply binds_app in Hb' as [Hb1' Hb2'];
    [ | rewrite length_map, <- (length_map snd wps1), <- Hkb1, length_map;
        reflexivity ].
  destruct (H1' hv hv' Hb1 Hb1') as (ws1 & ws1' & He1 & He1' & Hlw1 & Hr1).
  destruct (H2' hv hv' Hb2 Hb2') as (ws2 & ws2' & He2 & He2' & Hlw2 & Hr2).
  exists (ws1 ++ ws2), (ws1' ++ ws2').
  rewrite !eval_simples_app, He1, He2, He1', He2'.
  split; [ reflexivity | split; [ reflexivity | ] ].
  split; [ rewrite !length_app; lia | ].
  exact (vrel_app shk1 shk2 ws1 ws1' ws2 ws2' Hr1 Hlw1 Hr2).
Qed.

Lemma binds_of_bind (ps : list variable) (vs : list value) (cv hv : env) :
  NoDup ps -> bind_params ps vs cv = Some hv -> binds ps vs hv.
Proof.
  revert vs cv. induction ps as [ | p ps IH]; intros [ | v vs] cv Hnd Hb;
    simpl in Hb; try discriminate Hb; [ constructor | ].
  inversion Hnd as [ | p0 ps0 Hp Hnd1]; subst.
  constructor; [ | exact (IH vs _ Hnd1 Hb) ].
  rewrite (bind_params_other ps vs _ hv p Hb Hp).
  cbn. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma seg_wrap_both (sh_r sh_k : list pshape) (KS : list flambda_kind)
    (cps : list variable) (cmd : handler_metadata) (cargs : list simple)
    (wps : list (variable * flambda_kind)) (wmd : handler_metadata)
    (wargs : list simple) :
  seg_ok sh_r sh_k KS cps cargs wps wargs -> NoDup cps ->
  NoDup (map fst wps) -> hm_params_kinds cmd = KS ->
  hm_params_kinds wmd = map snd wps -> hm_params_subkinds wmd = [] ->
  wrap_both_ok sh_r sh_k cps cmd cargs (map fst wps) wmd wargs.
Proof.
  intros (Hl & Hc & H) Hnd Hnd' Hcmd Hwmd Hwsk.
  split; [ exact Hwsk | ]. intros vs vs' Hv Hk.
  rewrite Hcmd in Hk. apply kinds_eqb_eq in Hk.
  destruct (H vs vs' Hv Hk) as (Hk' & H').
  split; [ rewrite Hwmd, Hk'; apply kinds_eqb_refl | ].
  intros cv wv hv Hb.
  assert (Hlen : length (map fst wps) = length vs')
    by (rewrite length_map, <- (length_map snd wps), <- Hk', length_map;
        reflexivity).
  destruct (bind_params_some _ vs' wv Hlen) as (hv' & Hb').
  exists hv'. split; [ exact Hb' | ].
  intros ws Hws.
  destruct (H' hv hv' (binds_of_bind _ _ _ _ Hnd Hb)
              (binds_of_bind _ _ _ _ Hnd' Hb'))
    as (ws0 & ws' & Hws0 & Hws' & _ & Hr).
  rewrite Hws in Hws0. injection Hws0 as <-.
  exists ws'. split; [ exact Hws' | exact Hr ].
Qed.

Lemma seg_wrap_u (sh_r sh_k : list pshape) (KS : list flambda_kind)
    (cps : list variable) (wps : list (variable * flambda_kind))
    (wmd : handler_metadata) (wargs : list simple) :
  seg_ok sh_r sh_k KS cps (map Var cps) wps wargs -> NoDup cps ->
  NoDup (map fst wps) -> hm_params_kinds wmd = map snd wps ->
  hm_params_subkinds wmd = [] ->
  wrap_u_ok sh_r sh_k KS (map fst wps) wmd wargs.
Proof.
  intros (Hl & Hc & H) Hnd Hnd' Hwmd Hwsk.
  split; [ exact Hwsk | ]. intros vs vs' Hv Hk.
  destruct (H vs vs' Hv Hk) as (Hk' & H').
  split; [ rewrite Hwmd, Hk'; apply kinds_eqb_refl | ].
  intros wv.
  assert (Hlen : length cps = length vs)
    by (rewrite Hc, <- Hk, length_map; reflexivity).
  destruct (bind_params_some _ vs [] Hlen) as (hv & Hb).
  assert (Hlen' : length (map fst wps) = length vs')
    by (rewrite length_map, <- (length_map snd wps), <- Hk', length_map;
        reflexivity).
  destruct (bind_params_some _ vs' wv Hlen') as (hv' & Hb').
  destruct (H' hv hv' (binds_of_bind _ _ _ _ Hnd Hb)
              (binds_of_bind _ _ _ _ Hnd' Hb'))
    as (ws & ws' & Hws & Hws' & _ & Hr).
  rewrite (eval_bound_vars cps vs [] hv Hnd Hb) in Hws.
  injection Hws as <-.
  exists hv', ws'. split; [ exact Hb' | split; [ exact Hws' | exact Hr ] ].
Qed.

Lemma binds_fields_hold (fr : unboxed_fields) (v : value) (ws : list value)
    (hv : env) :
  fields_vals fr v ws -> binds (leaf_variables fr) ws hv ->
  fields_hold hv fr v.
Proof.
  intros Hws Hb path var k Hpath.
  destruct (upath_index _ _ _ _ _ Hpath) as (i & Hl & Hpi).
  destruct (forall2_nth_l _ _ _ i _ Hws Hpi) as (w' & Hw' & Hpw & Hk).
  exists w'. split; [ exact Hpw | split; [ | exact Hk ] ].
  apply (forall2_nth _ _ _ i var w' Hb); [ | exact Hw' ].
  unfold leaf_variables. rewrite nth_error_map.
  change (unboxed_field_leaves Value (Unboxed fr))
    with (fields_leaves fr) in Hl.
  rewrite Hl. reflexivity.
Qed.

(* The head of [freshen_decisions]. *)
Definition freshen_one (dk : decision * flambda_kind) (next : positive)
    : (wrapper_decision * flambda_kind) * positive :=
  match dk with
  | (Decided (Keep _), kind) | (Keep_unused _, kind) =>
    ((Wrapper_keep next, kind), Pos.succ next)
  | (Decided Delete, kind) => ((Wrapper_delete, kind), next)
  | (Unbox fields, kind) =>
    match rename_unboxed_field (Unboxed fields) next with
    | (Unboxed fields, next) => ((Wrapper_unbox fields, kind), next)
    | (Not_unboxed _, next) => ((Wrapper_delete, kind), next)
    end
  end.

Lemma freshen_cons (dk : decision * flambda_kind)
    (l : list (decision * flambda_kind)) (next : positive) :
  freshen_decisions (dk :: l) next
  = fst (freshen_one dk next)
    :: freshen_decisions l (snd (freshen_one dk next)).
Proof.
  destruct dk as [[[k | ] | fields | k] kind]; try reflexivity.
  cbn [freshen_decisions freshen_one].
  destruct (rename_unboxed_field (Unboxed fields) next) as [[x | f] n];
    reflexivity.
Qed.

(* The head of [wrapper_args_u]. *)
Definition wrapper_arg_u (tl : bool) (ad : decision) (fd : wrapper_decision)
    : option (list simple) :=
  match ad, fd with
  | Decided Delete, _ => Some []
  | Decided (Keep _), Wrapper_keep x | Keep_unused _, Wrapper_keep x =>
    Some [Var x]
  | Decided (Keep kind), Wrapper_delete
  | Keep_unused kind, (Wrapper_delete | Wrapper_unbox _) =>
    Some [Const (Poison kind)]
  | Decided (Keep kind), Wrapper_unbox _ =>
    if tl then Some [Const (Poison kind)] else None
  | Unbox _, (Wrapper_keep _ | Wrapper_delete) => None
  | Unbox fields_apply, Wrapper_unbox fields_func =>
    match
      unboxed_field_pairs Value (Unboxed fields_apply) (Unboxed fields_func)
    with
    | Some pairs => Some (map (fun '(_, x, _) => Var x) pairs)
    | None => None
    end
  end.

Lemma wrapper_args_u_cons (tl : bool) (ad : decision) (ads : list decision)
    (fd : wrapper_decision) (kind : flambda_kind)
    (func : list (wrapper_decision * flambda_kind)) :
  wrapper_args_u tl (ad :: ads) ((fd, kind) :: func)
  = match wrapper_args_u tl ads func, wrapper_arg_u tl ad fd with
    | Some rest, Some a => Some (a ++ rest)
    | _, _ => None
    end.
Proof.
  cbn [wrapper_args_u].
  destruct (wrapper_args_u tl ads func); [ | reflexivity ].
  destruct ad as [[k | ] | fa | k], fd as [x | | fu]; try reflexivity;
    cbn [wrapper_arg_u]; [ destruct tl; reflexivity | ].
  destruct (unboxed_field_pairs _ _ _); reflexivity.
Qed.

(* The decision of the first run for a result of the callee. *)
Definition cc_dec (r : pdesc) : param_decision :=
  if pd_keep r then Keep (pd_kind r) else Delete.

Lemma freshen_one_range (dk : decision * flambda_kind) (nu : positive) :
  NoDup (map fst (wrapper_params [fst (freshen_one dk nu)]))
  ∧ (∀ x : variable,
       x ∈ map fst (wrapper_params [fst (freshen_one dk nu)]) ->
       (nu <= x)%positive ∧ (x < snd (freshen_one dk nu))%positive)
  ∧ (nu <= snd (freshen_one dk nu))%positive.
Proof.
  destruct dk as [[[k | ] | fields | k] kind]; cbn [freshen_one fst snd].
  - cbn [wrapper_params flat_map map fst app].
    split; [ repeat constructor; intros [] | ].
    split; [ intros x [<- | []]; lia | lia ].
  - split; [ constructor | split; [ intros x [] | lia ] ].
  - destruct (rename_ok (Unboxed fields) Value nu)
      as ((f' & Hshape) & _ & Hnd & Hr & Hle).
    pose proof (Hshape fields eq_refl) as Hf.
    destruct (rename_unboxed_field (Unboxed fields) nu) as [u n'] eqn:Hu.
    cbn [fst snd] in Hf, Hnd, Hr, Hle. subst u.
    cbn [wrapper_params flat_map fst]. rewrite app_nil_r.
    split; [ exact Hnd | split; [ exact Hr | exact Hle ] ].
  - cbn [wrapper_params flat_map map fst app].
    split; [ repeat constructor; intros [] | ].
    split; [ intros x [<- | []]; lia | lia ].
Qed.

Lemma vrel_one_dead (vs vs' : list value) (kind : flambda_kind) :
  vrel [Psh_dead kind] vs vs' -> map kind_of_value vs = [kind] ->
  ∃ v v' : value, vs = [v] ∧ vs' = [v'] ∧ kind_of_value v' = kind.
Proof.
  intros Hv Hk.
  destruct (vrel_one_kept [Psh_dead kind] vs vs' kind (or_intror eq_refl) Hv
              Hk)
    as (v & v' & -> & -> & Hkv & _).
  exists v, v'. split; [ reflexivity | split; [ reflexivity | exact Hkv ] ].
Qed.

Lemma eval_one_var (n : variable) (v : value) (hv : env) :
  binds [n] [v] hv -> eval_simples hv [Var n] = Some [v].
Proof.
  intros Hb. inversion Hb as [ | p0 v0 l l' Hl _]; subst.
  cbn [eval_simples eval_simple]. rewrite Hl. reflexivity.
Qed.

Lemma freshen_one_unbox (fr : unboxed_fields) (kind : flambda_kind)
    (nu : positive) :
  ∃ fr' : unboxed_fields,
    fst (freshen_one (Unbox fr, kind) nu) = (Wrapper_unbox fr', kind)
    ∧ fields_paths fr' = fields_paths fr
    ∧ fst (rename_unboxed_field (Unboxed fr) nu) = Unboxed fr'.
Proof.
  destruct (freshen_unbox fr kind nu) as (fr' & Hf & Hp & _ & Hr).
  rewrite freshen_cons in Hf. injection Hf as Hf.
  exists fr'. split; [ exact Hf | split; [ exact Hp | exact Hr ] ].
Qed.

Lemma pairs_vars_in (fk fr : unboxed_fields)
    (pairs : list (variable * variable * flambda_kind)) (x : variable) :
  unboxed_field_pairs Value (Unboxed fk) (Unboxed fr) = Some pairs ->
  Var x ∈ map (fun '(_, x', _) => Var x') pairs ->
  x ∈ leaf_variables fr.
Proof.
  intros Hp Hx. apply in_map_iff in Hx as ([[y x'] k'] & Hx & Hin).
  injection Hx as ->.
  destruct (pairs_spec (Unboxed fk) Value (Unboxed fr) pairs Hp)
    as (H1 & _).
  destruct (H1 y x k' Hin) as (p & _ & Hpath).
  destruct (upath_index _ _ _ _ _ Hpath) as (i & Hl & _).
  unfold leaf_variables. apply in_map_iff. exists (x, k').
  split; [ reflexivity | ]. exact (nth_error_In _ _ Hl).
Qed.

Lemma inp_none (vs vs' : list value) :
  vrel [] vs vs' -> map kind_of_value vs = [] -> vs = [] ∧ vs' = [].
Proof.
  intros Hv Hk. destruct vs; [ | discriminate Hk ].
  apply vrel_none in Hv. split; [ reflexivity | exact Hv ].
Qed.

Lemma inp_plain (vs vs' : list value) (k : flambda_kind) :
  vrel [Psh_plain] vs vs' -> map kind_of_value vs = [k] ->
  ∃ v : value, vs = [v] ∧ vs' = [v] ∧ kind_of_value v = k.
Proof.
  intros Hv Hk. destruct (vrel_one_plain vs vs' k Hv Hk) as (v & -> & ->).
  exists v. cbn in Hk. injection Hk as Hk.
  split; [ reflexivity | split; [ reflexivity | exact Hk ] ].
Qed.

Lemma vrel_dead1 (k : flambda_kind) (v v' : value) :
  v' = v ∨ kind_of_value v' = k -> vrel [Psh_dead k] [v] [v'].
Proof. intros H. constructor; [ exact H | constructor ]. Qed.

Lemma vrel_plain1 (v : value) : vrel [Psh_plain] [v] [v].
Proof. constructor. constructor. Qed.

Lemma seg_one_ok (sh_r sh_k : list pshape) (KS : list flambda_kind)
    (cps : list variable) (cargs : list simple)
    (wps : list (variable * flambda_kind)) (wargs : list simple) :
  length sh_r = length KS -> length cps = length KS ->
  (∀ vs vs' : list value,
     vrel sh_r vs vs' -> map kind_of_value vs = KS ->
     map kind_of_value vs' = map snd wps
     ∧ ∀ hv hv' : env,
         binds cps vs hv -> binds (map fst wps) vs' hv' ->
         ∃ ws ws' : list value,
           eval_simples hv cargs = Some ws
           ∧ eval_simples hv' wargs = Some ws'
           ∧ length ws = length sh_k ∧ vrel sh_k ws ws') ->
  seg_ok sh_r sh_k KS cps cargs wps wargs.
Proof. intros H1 H2 H3. split; [ exact H1 | split; [ exact H2 | exact H3 ] ].
Qed.

Lemma result_seg (tl : bool) (r kp : pdesc) (n nu : positive) :
  result_ok kp r = true -> pdesc_ok kp = true -> pdesc_ok r = true ->
  ∃ wa : list simple,
    wrapper_arg_u tl (pd_decision kp)
      (fst (fst (freshen_one (pd_decision r, pd_kind r) nu)))
    = Some wa
    ∧ seg_ok (pshape_of r) (pshape_of kp) (kept_kind r)
        (if pd_keep r then [n] else [])
        (wrapper_args n [pd_keep kp] [(cc_dec r, pd_kind r)])
        (wrapper_params [fst (freshen_one (pd_decision r, pd_kind r) nu)]) wa
    ∧ (wa
       = map Var
           (map fst
              (wrapper_params
                 [fst (freshen_one (pd_decision r, pd_kind r) nu)]))
       -> pd_keep kp = pd_keep r)
    ∧ (∀ x : variable, Var x ∈ wa ->
         (nu <= x)%positive
         ∧ (x < snd (freshen_one (pd_decision r, pd_kind r) nu))%positive).
Proof.
  destruct r as [dr br kr], kp as [dk bk kk].
  unfold result_ok, pdesc_ok. cbn [pd_decision pd_keep pd_kind].
  intros Hres Hkp Hr.
  apply andb_true_iff in Hres as [Hkind Hres]. apply kind_eqb_eq in Hkind.
  subst kk.
  apply andb_true_iff in Hkp as [_ Hkp].
  apply andb_true_iff in Hr as [_ Hr].
  unfold cc_dec, pshape_of, kept_kind. cbn [pd_decision pd_keep pd_kind].
  destruct dr as [[k1 | ] | fr | k1]; destruct dk as [[k2 | ] | fk | k2];
    try discriminate Hres.
  - (* Kept by both. *)
    apply andb_true_iff in Hr as [-> Hk1]. apply kind_eqb_eq in Hk1.
    apply andb_true_iff in Hkp as [-> Hk2]. apply kind_eqb_eq in Hk2.
    subst k1 k2. cbn [freshen_one fst snd wrapper_arg_u].
    exists [Var nu]. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    split; [ | split; [ intros _; reflexivity | ] ].
    + apply seg_one_ok; [ reflexivity | reflexivity | ].
      intros vs vs' Hv Hk.
      destruct (inp_plain vs vs' kr Hv Hk) as (v & -> & -> & Hkv).
      split; [ cbn; rewrite Hkv; reflexivity | ].
      intros hv hv' Hb Hb'. exists [v], [v].
      rewrite (eval_one_var n v hv Hb), (eval_one_var nu v hv' Hb').
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ reflexivity | apply vrel_plain1 ].
    + intros x [Hx | []]. injection Hx as <-. lia.
  - (* Deleted by the continuation, kept by the callee. *)
    apply andb_true_iff in Hr as [-> Hk1]. apply kind_eqb_eq in Hk1.
    apply negb_true_iff in Hkp. subst bk k1.
    cbn [freshen_one fst snd wrapper_arg_u].
    exists []. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    split; [ | split; [ intros H; discriminate H | intros x [] ] ].
    apply seg_one_ok; [ reflexivity | reflexivity | ].
    intros vs vs' Hv Hk.
    destruct (inp_plain vs vs' kr Hv Hk) as (v & -> & -> & Hkv).
    split; [ cbn; rewrite Hkv; reflexivity | ].
    intros hv hv' Hb Hb'. exists [], [].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | constructor ].
  - (* Kept with no use by the continuation, kept by the callee. *)
    apply andb_true_iff in Hr as [-> Hk1]. apply kind_eqb_eq in Hk1.
    apply andb_true_iff in Hkp as [-> Hk2]. apply kind_eqb_eq in Hk2.
    subst k1 k2. cbn [freshen_one fst snd wrapper_arg_u].
    eexists. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    split; [ | split; [ intros _; reflexivity | ] ].
    + apply seg_one_ok; [ reflexivity | reflexivity | ].
      intros vs vs' Hv Hk.
      destruct (inp_plain vs vs' kr Hv Hk) as (v & -> & -> & Hkv).
      split; [ cbn; rewrite Hkv; reflexivity | ].
      intros hv hv' Hb Hb'.
      exists [v], [v].
      rewrite (eval_one_var n v hv Hb).
      split; [ reflexivity | ].
      split; [ exact (eval_one_var nu v hv' Hb') | ].
      split; [ reflexivity | apply vrel_dead1 ].
      left; reflexivity.
    + intros x Hx.
      destruct Hx as [Hx | []]. injection Hx as <-. lia.
  - (* Kept by the continuation, deleted by the callee. *)
    apply negb_true_iff in Hr.
    apply andb_true_iff in Hkp as [-> Hk2]. apply kind_eqb_eq in Hk2.
    subst br k2. cbn [freshen_one fst snd wrapper_arg_u].
    exists [Const (Poison kr)]. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    split; [ | split; [ intros H; discriminate H
                      | intros x [Hx | []]; discriminate Hx ] ].
    apply seg_one_ok; [ reflexivity | reflexivity | ].
    intros vs vs' Hv Hk.
    destruct (inp_none vs vs' Hv Hk) as [-> ->].
    split; [ reflexivity | ].
    intros hv hv' _ _.
    exists [Value_const (Poison kr)], [Value_const (Poison kr)].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | apply vrel_plain1 ].
  - (* Deleted by both. *)
    apply negb_true_iff in Hr. apply negb_true_iff in Hkp. subst br bk.
    cbn [freshen_one fst snd wrapper_arg_u].
    exists []. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    split; [ | split; [ intros _; reflexivity | intros x [] ] ].
    apply seg_one_ok; [ reflexivity | reflexivity | ].
    intros vs vs' Hv Hk.
    destruct (inp_none vs vs' Hv Hk) as [-> ->].
    split; [ reflexivity | ].
    intros hv hv' _ _. exists [], [].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | constructor ].
  - (* Kept with no use by the continuation, deleted by the callee. *)
    apply negb_true_iff in Hr.
    apply andb_true_iff in Hkp as [-> Hk2]. apply kind_eqb_eq in Hk2.
    subst br k2. cbn [freshen_one fst snd wrapper_arg_u].
    exists [Const (Poison kr)]. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    split; [ | split; [ intros H; discriminate H
                      | intros x [Hx | []]; discriminate Hx ] ].
    apply seg_one_ok; [ reflexivity | reflexivity | ].
    intros vs vs' Hv Hk.
    destruct (inp_none vs vs' Hv Hk) as [-> ->].
    split; [ reflexivity | ].
    intros hv hv' _ _.
    exists [Value_const (Poison kr)], [Value_const (Poison kr)].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | apply vrel_dead1; left; reflexivity ].
  - (* Unboxed by the callee, deleted by the continuation. *)
    apply negb_true_iff in Hkp. subst bk.
    apply andb_true_iff in Hr as [Hv0 Hr]. apply kind_eqb_eq in Hv0.
    subst kr.
    destruct (freshen_one_unbox fr Value nu) as (fr' & Hf & Hpaths & Hren).
    pose proof (freshen_one_range (Unbox fr, Value) nu) as (_ & Hrange & _).
    rewrite Hf in Hrange |- *. cbn [fst wrapper_arg_u].
    exists []. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    rewrite app_nil_r.
    split; [ | split; [ | intros x [] ] ].
    + destruct br.
      * apply seg_one_ok; [ reflexivity | reflexivity | ].
        intros vs vs' Hv Hk.
        destruct (vrel_one_unboxed fr vs vs' Value Hv Hk) as (v & -> & Hws).
        split; [ rewrite ?app_nil_r;
                 exact (unbox_kinds fr fr' v vs' Hpaths Hws) | ].
        intros hv hv' _ _. exists [], [].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | constructor ].
      * cbn [orb] in Hr.
        assert (Hl : fields_leaves fr' = []).
        { apply leaves_of_paths_nil. rewrite Hpaths.
          exact (no_leaves_paths fr Hr). }
        rewrite Hl.
        apply seg_one_ok; [ reflexivity | reflexivity | ].
        intros vs vs' Hv Hk.
        destruct (inp_none vs vs' Hv Hk) as [-> ->].
        split; [ reflexivity | ].
        intros hv hv' _ _. exists [], [].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | constructor ].
    + intros Heq. destruct br; [ | reflexivity ]. exfalso.
      rewrite andb_true_r in Hres. apply negb_true_iff in Hres.
      assert (Hl : fields_leaves fr' <> []).
      { intros Hl.
        assert (Hp' : fields_paths fr' = [])
          by (pose proof (Forall2_length (fields_leaves_paths fr')) as Hl';
              rewrite Hl in Hl'; destruct (fields_paths fr');
              [ reflexivity | discriminate Hl' ]).
        rewrite Hpaths in Hp'. apply leaves_of_paths_nil in Hp'.
        unfold no_leaves in Hres. rewrite Hp' in Hres. discriminate Hres. }
      destruct (fields_leaves fr'); [ exact (Hl eq_refl) | ].
      discriminate Heq.
  - (* Unboxed by both. *)
    apply andb_true_iff in Hr as [Hv0 Hr]. apply kind_eqb_eq in Hv0.
    subst kr.
    apply andb_true_iff in Hkp as [_ Hkp].
    apply andb_true_iff in Hres as [Halike Hres].
    unfold fields_alike in Halike.
    destruct (unboxed_field_pairs Value (Unboxed fk) (Unboxed fr))
      as [pairs0 | ] eqn:Hp0; [ | discriminate Halike ].
    destruct (freshen_one_unbox fr Value nu) as (fr' & Hf & Hpaths & Hren).
    pose proof (freshen_one_range (Unbox fr, Value) nu) as (_ & Hrange & _).
    destruct (pairs_rename (Unboxed fk) Value (Unboxed fr) nu pairs0 Hp0)
      as (pairs & Hp). rewrite Hren in Hp.
    rewrite Hf in Hrange |- *. cbn [fst wrapper_arg_u]. rewrite Hp.
    eexists. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst] in Hrange |- *.
    rewrite ?app_nil_r in Hrange |- *.
    split; [ | split ].
    + destruct br, bk.
      * apply seg_one_ok; [ reflexivity | reflexivity | ].
        intros vs vs' Hv Hk.
        destruct (vrel_one_unboxed fr vs vs' Value Hv Hk) as (v & -> & Hws).
        split; [ rewrite ?app_nil_r;
                 exact (unbox_kinds fr fr' v vs' Hpaths Hws) | ].
        intros hv hv' Hb Hb'. rewrite ?app_nil_r in Hb'.
        assert (Hws' : fields_vals fr' v vs')
          by (unfold fields_vals; rewrite Hpaths; exact Hws).
        destruct (pairs_fields_vals hv' fk fr' v pairs Hp
                    (binds_fields_hold fr' v vs' hv' Hws' Hb'))
          as (ws'' & Hev & Hfk).
        exists [v], ws''.
        rewrite (eval_one_var n v hv Hb).
        split; [ reflexivity | split; [ exact Hev | ] ].
        split; [ reflexivity | ].
        rewrite <- (app_nil_r ws'').
        apply vrel_unboxed; [ exact Hfk | constructor ].
      * cbn [orb] in Hkp.
        rewrite (pairs_no_leaves fk fr' pairs Hp (no_leaves_nil fk Hkp)).
        apply seg_one_ok; [ reflexivity | reflexivity | ].
        intros vs vs' Hv Hk.
        destruct (vrel_one_unboxed fr vs vs' Value Hv Hk) as (v & -> & Hws).
        split; [ rewrite ?app_nil_r;
                 exact (unbox_kinds fr fr' v vs' Hpaths Hws) | ].
        intros hv hv' _ _. exists [], [].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | constructor ].
      * cbn [orb] in Hr. rewrite Hr in Hres. discriminate Hres.
      * cbn [orb] in Hkp, Hr.
        rewrite (pairs_no_leaves fk fr' pairs Hp (no_leaves_nil fk Hkp)).
        assert (Hl : fields_leaves fr' = []).
        { apply leaves_of_paths_nil. rewrite Hpaths.
          exact (no_leaves_paths fr Hr). }
        rewrite Hl.
        apply seg_one_ok; [ reflexivity | reflexivity | ].
        intros vs vs' Hv Hk.
        destruct (inp_none vs vs' Hv Hk) as [-> ->].
        split; [ reflexivity | ].
        intros hv hv' _ _. exists [], [].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | constructor ].
    + intros Heq. destruct br, bk; try reflexivity; exfalso.
      * cbn [orb] in Hkp.
        rewrite (pairs_no_leaves fk fr' pairs Hp (no_leaves_nil fk Hkp))
          in Heq.
        cbn [Bool.eqb orb] in Hres. rewrite orb_false_r in Hres.
        apply negb_true_iff in Hres.
        assert (Hl : fields_leaves fr' <> []).
        { intros Hl.
          assert (Hp' : fields_paths fr' = [])
            by (pose proof (Forall2_length (fields_leaves_paths fr')) as Hl';
                rewrite Hl in Hl'; destruct (fields_paths fr');
                [ reflexivity | discriminate Hl' ]).
          rewrite Hpaths in Hp'. apply leaves_of_paths_nil in Hp'.
          unfold no_leaves in Hres. rewrite Hp' in Hres.
          discriminate Hres. }
        destruct (fields_leaves fr'); [ exact (Hl eq_refl) | ].
        discriminate Heq.
      * cbn [orb] in Hr. rewrite Hr in Hres. discriminate Hres.
    + intros x Hx.
      exact (Hrange x (pairs_vars_in fk fr' pairs x Hp Hx)).
  - (* Unboxed by the callee, kept with no use by the continuation. *)
    apply andb_true_iff in Hkp as [-> Hk2]. apply kind_eqb_eq in Hk2.
    subst k2.
    apply andb_true_iff in Hr as [Hv0 Hr]. apply kind_eqb_eq in Hv0.
    subst kr.
    destruct (freshen_one_unbox fr Value nu) as (fr' & Hf & Hpaths & Hren).
    rewrite Hf. cbn [fst wrapper_arg_u].
    exists [Const (Poison Value)]. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    rewrite app_nil_r.
    split; [ | split; [ | intros x [Hx | []]; discriminate Hx ] ].
    + destruct br.
      * apply seg_one_ok; [ reflexivity | reflexivity | ].
        intros vs vs' Hv Hk.
        destruct (vrel_one_unboxed fr vs vs' Value Hv Hk) as (v & -> & Hws).
        split; [ rewrite ?app_nil_r;
                 exact (unbox_kinds fr fr' v vs' Hpaths Hws) | ].
        intros hv hv' Hb _. exists [v], [Value_const (Poison Value)].
        rewrite (eval_one_var n v hv Hb).
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | apply vrel_dead1; right; reflexivity ].
      * cbn [orb] in Hr.
        assert (Hl : fields_leaves fr' = []).
        { apply leaves_of_paths_nil. rewrite Hpaths.
          exact (no_leaves_paths fr Hr). }
        rewrite Hl.
        apply seg_one_ok; [ reflexivity | reflexivity | ].
        intros vs vs' Hv Hk.
        destruct (inp_none vs vs' Hv Hk) as [-> ->].
        split; [ reflexivity | ].
        intros hv hv' _ _.
        exists [Value_const (Poison Value)], [Value_const (Poison Value)].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | apply vrel_dead1; left; reflexivity ].
    + intros Heq. destruct (fields_leaves fr'); discriminate Heq.
  - (* Kept with no use by the callee, deleted by the continuation. *)
    apply andb_true_iff in Hr as [-> Hk1]. apply kind_eqb_eq in Hk1.
    apply negb_true_iff in Hkp. subst bk k1.
    cbn [freshen_one fst snd wrapper_arg_u].
    exists []. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    split; [ | split; [ intros H; discriminate H | intros x [] ] ].
    apply seg_one_ok; [ reflexivity | reflexivity | ].
    intros vs vs' Hv Hk.
    destruct (vrel_one_dead vs vs' kr Hv Hk) as (v & v' & -> & -> & Hkv).
    split; [ cbn; rewrite Hkv; reflexivity | ].
    intros hv hv' Hb Hb'. exists [], [].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | constructor ].
  - (* Kept with no use by both. *)
    apply andb_true_iff in Hr as [-> Hk1]. apply kind_eqb_eq in Hk1.
    apply andb_true_iff in Hkp as [-> Hk2]. apply kind_eqb_eq in Hk2.
    subst k1 k2. cbn [freshen_one fst snd wrapper_arg_u].
    eexists. split; [ reflexivity | ].
    cbn [wrapper_args wrapper_params flat_map app map fst].
    split; [ | split; [ intros _; reflexivity | ] ].
    + apply seg_one_ok; [ reflexivity | reflexivity | ].
      intros vs vs' Hv Hk.
      destruct (vrel_one_dead vs vs' kr Hv Hk) as (v & v' & -> & -> & Hkv).
      split; [ cbn; rewrite Hkv; reflexivity | ].
      intros hv hv' Hb Hb'.
      exists [v], [v'].
      rewrite (eval_one_var n v hv Hb).
      split; [ reflexivity | ].
      split; [ exact (eval_one_var nu v' hv' Hb') | ].
      split; [ reflexivity | apply vrel_dead1; right ].
      exact Hkv.
    + intros x Hx.
      destruct Hx as [Hx | []]. injection Hx as <-. lia.
Qed.

Lemma freshen_range (l : list (decision * flambda_kind)) :
  ∀ nu : positive,
    NoDup (map fst (wrapper_params (freshen_decisions l nu)))
    ∧ ∀ x : variable,
        x ∈ map fst (wrapper_params (freshen_decisions l nu)) ->
        (nu <= x)%positive.
Proof.
  induction l as [ | dk l IH]; intros nu.
  - split; [ constructor | intros x [] ].
  - rewrite freshen_cons.
    destruct (freshen_one_range dk nu) as (Hnd1 & Hr1 & Hle1).
    destruct (IH (snd (freshen_one dk nu))) as (Hnd2 & Hr2).
    assert (Hsplit :
      wrapper_params (fst (freshen_one dk nu)
                      :: freshen_decisions l (snd (freshen_one dk nu)))
      = wrapper_params [fst (freshen_one dk nu)]
        ++ wrapper_params (freshen_decisions l (snd (freshen_one dk nu))))
      by (unfold wrapper_params; cbn [flat_map]; rewrite app_nil_r;
          reflexivity).
    rewrite Hsplit, map_app.
    split.
    + apply NoDup_app; [ exact Hnd1 | exact Hnd2 | ].
      intros x Hx Hx'. destruct (Hr1 x Hx) as [_ Hlt].
      specialize (Hr2 x Hx'). lia.
    + intros x Hx. apply in_app_or in Hx as [Hx | Hx];
        [ destruct (Hr1 x Hx); lia | specialize (Hr2 x Hx); lia ].
Qed.

Lemma split_vars (a : list simple) :
  ∀ (b : list simple) (c d : list variable) (h : positive),
    a ++ b = map Var (c ++ d) ->
    (∀ x, Var x ∈ a -> (x < h)%positive) ->
    (∀ x, x ∈ c -> (x < h)%positive) ->
    (∀ x, Var x ∈ b -> (h <= x)%positive) ->
    (∀ x, x ∈ d -> (h <= x)%positive) ->
    a = map Var c ∧ b = map Var d.
Proof.
  induction a as [ | s a IH]; intros b c d h Heq Ha Hc Hb Hd.
  - destruct c as [ | x c]; [ split; [ reflexivity | exact Heq ] | ].
    exfalso. cbn [app map] in Heq.
    assert (Hx : Var x ∈ b) by (rewrite Heq; left; reflexivity).
    specialize (Hb x Hx). specialize (Hc x (or_introl eq_refl)). lia.
  - destruct c as [ | x c].
    + exfalso. cbn [app map] in Heq.
      destruct d as [ | y d]; [ discriminate Heq | ].
      cbn [map] in Heq. injection Heq as -> _.
      specialize (Ha y (or_introl eq_refl)).
      specialize (Hd y (or_introl eq_refl)). lia.
    + cbn [app map] in Heq. injection Heq as -> Heq.
      destruct (IH b c d h Heq (fun y Hy => Ha y (or_intror Hy))
                  (fun y Hy => Hc y (or_intror Hy)) Hb Hd)
        as [-> ->].
      split; reflexivity.
Qed.

Lemma results_seg (tl : bool) (kps : list pdesc) :
  ∀ (rds : list pdesc) (n nu : positive),
    forallb2 result_ok kps rds = true -> forallb pdesc_ok kps = true ->
    forallb pdesc_ok rds = true ->
    ∃ wargs : list simple,
      wrapper_args_u tl (map pd_decision kps)
        (freshen_decisions (combine (map pd_decision rds) (map pd_kind rds))
           nu)
      = Some wargs
      ∧ seg_ok (flat_map pshape_of rds) (flat_map pshape_of kps)
          (flat_map kept_kind rds)
          (FlambdaFreshQuantifierSpec.names_from
             (length (kept_kinds (map cc_dec rds))) n)
          (wrapper_args n (map pd_keep kps)
             (combine (map cc_dec rds) (map pd_kind rds)))
          (wrapper_params
             (freshen_decisions
                (combine (map pd_decision rds) (map pd_kind rds)) nu))
          wargs
      ∧ (wargs
         = map Var
             (map fst
                (wrapper_params
                   (freshen_decisions
                      (combine (map pd_decision rds) (map pd_kind rds)) nu)))
         -> map pd_keep kps = map pd_keep rds)
      ∧ (∀ x : variable, Var x ∈ wargs -> (nu <= x)%positive).
Proof.
  induction kps as [ | kp kps IH]; intros [ | r rds] n nu Hres Hkps Hrds;
    try discriminate Hres.
  - exists []. split; [ reflexivity | ].
    split; [ exact seg_nil | ].
    split; [ intros _; reflexivity | intros x [] ].
  - cbn [forallb2] in Hres. apply andb_true_iff in Hres as [Hr1 Hres].
    cbn [forallb] in Hkps, Hrds.
    apply andb_true_iff in Hkps as [Hkp Hkps].
    apply andb_true_iff in Hrds as [Hrd Hrds].
    cbn [map combine]. rewrite freshen_cons.
    set (hd := freshen_one (pd_decision r, pd_kind r) nu) in *.
    destruct (result_seg tl r kp n nu Hr1 Hkp Hrd)
      as (wa & Hwa & Hseg & Hkeep & Hra).
    fold hd in Hwa, Hseg, Hkeep, Hra.
    set (n' := if pd_keep r then Pos.succ n else n).
    destruct (IH rds n' (snd hd) Hres Hkps Hrds)
      as (wr & Hwr & Hsegr & Hkeepr & Hrr).
    destruct (fst hd) as [fd kind] eqn:Hfd.
    rewrite wrapper_args_u_cons, Hwr. cbn [fst] in Hwa. rewrite Hwa.
    exists (wa ++ wr). split; [ reflexivity | ].
    assert (Hwp :
      wrapper_params ((fd, kind) :: freshen_decisions
                        (combine (map pd_decision rds) (map pd_kind rds))
                        (snd hd))
      = wrapper_params [(fd, kind)]
        ++ wrapper_params
             (freshen_decisions
                (combine (map pd_decision rds) (map pd_kind rds)) (snd hd)))
      by (unfold wrapper_params; cbn [flat_map]; rewrite app_nil_r;
          reflexivity).
    assert (Hnames :
      FlambdaFreshQuantifierSpec.names_from
        (length (kept_kinds (cc_dec r :: map cc_dec rds))) n
      = (if pd_keep r then [n] else [])
        ++ FlambdaFreshQuantifierSpec.names_from
             (length (kept_kinds (map cc_dec rds))) n')
      by (unfold n', cc_dec; cbn [map kept_kinds flat_map];
          destruct (pd_keep r); reflexivity).
    assert (Hcargs :
      wrapper_args n (pd_keep kp :: map pd_keep kps)
        ((cc_dec r, pd_kind r)
         :: combine (map cc_dec rds) (map pd_kind rds))
      = wrapper_args n [pd_keep kp] [(cc_dec r, pd_kind r)]
        ++ wrapper_args n' (map pd_keep kps)
             (combine (map cc_dec rds) (map pd_kind rds)))
      by (unfold n', cc_dec; destruct (pd_keep r), (pd_keep kp);
          cbn [wrapper_args app]; rewrite ?app_nil_r; reflexivity).
    split; [ | split ].
    + cbn [flat_map]. rewrite Hwp, Hnames.
      rewrite Hcargs.
      exact (seg_app _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hseg Hsegr).
    + intros Heq. rewrite Hwp, (map_app fst) in Heq.
      destruct (freshen_one_range (pd_decision r, pd_kind r) nu)
        as (_ & Hrange1 & _).
      fold hd in Hrange1. rewrite Hfd in Hrange1.
      destruct (freshen_range (combine (map pd_decision rds)
                                 (map pd_kind rds)) (snd hd))
        as (_ & Hrange2).
      destruct (split_vars wa wr _ _ (snd hd) Heq
                  (fun x Hx => proj2 (Hra x Hx))
                  (fun x Hx => proj2 (Hrange1 x Hx)) Hrr Hrange2)
        as [H1 H2].
      cbn [map]. rewrite (Hkeep H1), (Hkeepr H2). reflexivity.
    + intros x Hx. apply in_app_or in Hx as [Hx | Hx];
        [ exact (proj1 (Hra x Hx)) | ].
      destruct (freshen_one_range (pd_decision r, pd_kind r) nu)
        as (_ & _ & Hle).
      fold hd in Hle. specialize (Hrr x Hx). lia.
Qed.

Lemma cc_nowrap_iff (kps rds : list pdesc) :
  length kps = length rds ->
  forallb (fun '(keep, (decision, _)) => Bool.eqb keep (is_kept decision))
    (combine (map pd_keep kps) (combine (map cc_dec rds) (map pd_kind rds)))
  = forallb2 Bool.eqb (map pd_keep kps) (map pd_keep rds).
Proof.
  revert rds. induction kps as [ | kp kps IH]; intros [ | r rds] Hl;
    simpl in Hl; try discriminate Hl; [ reflexivity | ].
  cbn [map combine forallb forallb2]. rewrite IH by lia.
  unfold cc_dec. destruct (pd_keep r); reflexivity.
Qed.

Lemma cc_nowrap_args (kps : list pdesc) :
  ∀ (rds : list pdesc) (n : positive),
    forallb2 Bool.eqb (map pd_keep kps) (map pd_keep rds) = true ->
    wrapper_args n (map pd_keep kps)
      (combine (map cc_dec rds) (map pd_kind rds))
    = map Var
        (FlambdaFreshQuantifierSpec.names_from
           (length (kept_kinds (map cc_dec rds))) n).
Proof.
  induction kps as [ | kp kps IH]; intros [ | r rds] n H;
    cbn [map forallb2] in H; try discriminate H; [ reflexivity | ].
  apply andb_true_iff in H as [Hb H]. apply Bool.eqb_prop in Hb.
  cbn [map combine]. rewrite Hb.
  assert (Hc : cc_dec r = if pd_keep r then Keep (pd_kind r) else Delete)
    by reflexivity.
  rewrite Hc. specialize (IH rds). unfold kept_kinds in IH |- *.
  destruct (pd_keep r); cbn [wrapper_args app flat_map length app];
    rewrite IH by exact H; reflexivity.
Qed.

Lemma kept_kinds_cc (rds : list pdesc) :
  kept_kinds (map cc_dec rds) = flat_map kept_kind rds.
Proof.
  induction rds as [ | r rds IH]; [ reflexivity | ].
  unfold kept_kinds in *. cbn [map flat_map]. rewrite IH.
  unfold cc_dec, kept_kind. destruct (pd_keep r); reflexivity.
Qed.

Lemma kept_kinds_same (kps : list pdesc) :
  ∀ rds : list pdesc,
    forallb2 result_ok kps rds = true ->
    map pd_keep kps = map pd_keep rds ->
    flat_map kept_kind kps = flat_map kept_kind rds.
Proof.
  induction kps as [ | kp kps IH]; intros [ | r rds] Hres Hk;
    try discriminate Hres; [ reflexivity | ].
  cbn [forallb2] in Hres. apply andb_true_iff in Hres as [Hr Hres].
  unfold result_ok in Hr. apply andb_true_iff in Hr as [Hkind _].
  apply kind_eqb_eq in Hkind. cbn [map] in Hk. injection Hk as Hk1 Hk.
  cbn [flat_map]. rewrite (IH rds Hres Hk). unfold kept_kind.
  rewrite Hk1, Hkind. reflexivity.
Qed.

(* ======================================================================== *)
(* Several groups of arguments, several results                             *)
(* ======================================================================== *)

Lemma forallb2_length {A B : Type} (p : A -> B -> bool) (l : list A) :
  ∀ l' : list B, forallb2 p l l' = true -> length l = length l'.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] H; simpl in H;
    try discriminate H; [ reflexivity | ].
  apply andb_true_iff in H as [_ H]. simpl. f_equal. exact (IH l' H).
Qed.

Lemma forallb2_eqb_refl (l : list bool) : forallb2 Bool.eqb l l = true.
Proof.
  induction l as [ | b l IH]; [ reflexivity | ]. cbn [forallb2].
  rewrite IH, eqb_reflx. reflexivity.
Qed.

Lemma map_fst_combine_eq {A B : Type} (l : list A) :
  ∀ l' : list B, length l = length l' -> map fst (combine l l') = l.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] H; simpl in H;
    try discriminate H; [ reflexivity | ].
  simpl. rewrite IH by lia. reflexivity.
Qed.

(* The mask of the first run for the [n] results of a call returning to a
   continuation described by [kps]. *)
Lemma apply_decisions_kps (C : finite_map (list pdesc * bool))
    (en : rebuild_env) (enu : unbox_env) (k : continuation) (kps : list pdesc)
    (h : bool) :
  cinv C en enu -> lookup k C = Some (kps, h) ->
  apply_decisions en k (length kps) = map pd_keep kps
  ∧ (lookup k (env_conts en) = Some (map pd_keep kps)
     ∨ lookup k (env_conts en) = None ∧ forallb pd_keep kps = true).
Proof.
  intros Hci HCk.
  destruct (ci_mask _ _ _ Hci k) as [Hm | (Hm & ps & h' & HC & Hall)].
  - rewrite HCk in Hm. cbn [option_map fst] in Hm.
    unfold apply_decisions. rewrite Hm.
    rewrite firstn_app, length_map, Nat.sub_diag, firstn_O, app_nil_r.
    rewrite <- (length_map pd_keep kps), firstn_all.
    split; [ reflexivity | left; reflexivity ].
  - rewrite HCk in HC. injection HC as <- _.
    unfold apply_decisions. rewrite Hm. cbn [app].
    rewrite firstn_all2 by (rewrite repeat_length; lia).
    split; [ | right; split; [ reflexivity | exact Hall ] ].
    clear -Hall. induction kps as [ | p kps IH]; [ reflexivity | ].
    cbn [forallb] in Hall. apply andb_true_iff in Hall as [Hp Hall].
    cbn [length repeat map]. rewrite Hp, (IH Hall). reflexivity.
Qed.

(* The results of a call, as the two rebuilds make them: the continuation
   described by [kps], the callee by [rds], each rebuild with a wrapper or
   not. *)
Lemma uc_result_results (K : finite_map kshape)
    (ap0 ap_cc ap_u : apply_expr) (cs : code_shape) (k : continuation)
    (kps rds : list pdesc) (en : rebuild_env) (enu : unbox_env)
    (fd_u : list (decision * flambda_kind)) :
  call_rel unboxed dead CSf RCHf OVf REMf ap_cc ap_u cs ->
  exn_call_rel unboxed dead K (cs_exn cs) ap_cc ap_u ->
  ap_continuation ap_cc = k -> ap_continuation ap_u = k ->
  ap_exn_continuation ap_cc = ap_exn_continuation ap0 ->
  ap_continuation ap0 = k ->
  ff <> ap_exn_continuation ap_cc -> ffu <> ap_exn_continuation ap_cc ->
  lookup k K
  = Some (Kshape (flat_map pshape_of kps) (flat_map kept_kind kps) true) ->
  cs_results cs = flat_map pshape_of rds ->
  cs_result_kinds cs = flat_map kept_kind rds ->
  forallb2 result_ok kps rds = true ->
  forallb pdesc_ok kps = true -> forallb pdesc_ok rds = true ->
  apply_decisions en k (length rds) = map pd_keep kps ->
  lookup k (ue_decisions enu) = Some (map pd_decision kps) ->
  freshen_decisions fd_u ffu
  = freshen_decisions (combine (map pd_decision rds) (map pd_kind rds)) ffu ->
  uc_result K (free_continuations_of_expr (Apply ap0))
    (make_apply_wrapper_cc ff en ap_cc
       (combine (map cc_dec rds) (map pd_kind rds)))
    (make_apply_wrapper_u true ffu enu ap_u fd_u false).
Proof.
  intros Hcall Hexn Hk Hk' He Hk0 Hffe Hffue HKk Hcsr Hcsk Hres Hkps Hrds
    Happ Hdec Hfd.
  assert (Hlen : length kps = length rds)
    by exact (forallb2_length _ _ _ Hres).
  assert (Hlfd : length (combine (map cc_dec rds) (map pd_kind rds))
                 = length rds)
    by (rewrite length_combine, !length_map; lia).
  unfold make_apply_wrapper_cc, make_apply_wrapper_u. cbv zeta.
  cbn [andb].
  rewrite Hlfd, Hk, Happ, Hk', Hdec, Hfd. cbn [option_map].
  set (func := freshen_decisions
                 (combine (map pd_decision rds) (map pd_kind rds)) ffu).
  destruct (results_seg true kps rds ff ffu Hres Hkps Hrds)
    as (wargs & Hwa & Hseg & Hkeep & _).
  fold func in Hwa, Hseg, Hkeep.
  destruct (freshen_range (combine (map pd_decision rds) (map pd_kind rds))
              ffu)
    as (Hnd & _).
  fold func in Hnd.
  rewrite Hwa, (cc_nowrap_iff kps rds Hlen).
  destruct (forallb2 Bool.eqb (map pd_keep kps) (map pd_keep rds)) eqn:Hnw.
  - (* No wrapper in the first run. *)
    pose proof (forallb2_eqb_eq _ _ Hnw) as Hmask.
    rewrite (cc_nowrap_args kps rds ff Hnw) in Hseg.
    assert (HKk' : lookup k K
                   = Some (Kshape (flat_map pshape_of kps)
                             (cs_result_kinds cs) true))
      by (rewrite HKk, Hcsk, (kept_kinds_same kps rds Hres Hmask);
          reflexivity).
    rewrite <- Hcsr, <- Hcsk in Hseg.
    pose proof (seg_wrap_u _ _ _ _ _
                  (make_handler_metadata (map snd (wrapper_params func))) _
                  Hseg (FlambdaReaperRebuildProofs.names_from_nodup _ _) Hnd
                  eq_refl eq_refl)
      as Hwu.
    destruct (args_are_params wargs (map fst (wrapper_params func))) eqn:Hap.
    + apply (uc_result_call K ap0 ap_cc ap_u cs k _ Hcall Hexn Hk Hk' He Hk0
               HKk').
      apply args_are_params_eq in Hap.
      intros vs vs' Hv Hkv.
      destruct (proj2 Hwu vs vs' Hv Hkv) as (_ & H).
      destruct (H []) as (hv & vs'' & Hb & Hev & Hr).
      rewrite Hap, (eval_bound_vars _ vs' [] hv Hnd Hb) in Hev.
      injection Hev as <-. exact Hr.
    + eapply (uc_result_call_wrap K ap0 ap_cc (with_continuation ap_u ffu) cs
                k _ ffu _ _ _ (call_rel_with _ _ _ k ffu Hcall) Hexn Hk
                eq_refl Hffue He Hk0 HKk').
      exact Hwu.
  - (* Wrappers in both runs. *)
    destruct (args_are_params wargs (map fst (wrapper_params func))) eqn:Hap.
    { exfalso. apply args_are_params_eq in Hap.
      rewrite (Hkeep Hap), forallb2_eqb_refl in Hnw. discriminate Hnw. }
    assert (Hfst : map fst (combine (map cc_dec rds) (map pd_kind rds))
                   = map cc_dec rds)
      by (apply map_fst_combine_eq; rewrite !length_map; reflexivity).
    rewrite Hfst.
    eapply (uc_result_call_wraps K ap0 _ _ cs k (flat_map pshape_of kps) _ ff
              _ _ _ ffu _ _ _ (call_rel_with _ _ _ ff ffu Hcall) Hexn eq_refl
              eq_refl Hffe Hffue He Hk0 HKk).
    + cbn [hm_params_kinds make_handler_metadata].
      rewrite kept_kinds_cc, Hcsk. reflexivity.
    + rewrite Hcsr.
      apply (seg_wrap_both _ _ (flat_map kept_kind rds)); [ | | | | | ].
      * exact Hseg.
      * apply FlambdaReaperRebuildProofs.names_from_nodup.
      * exact Hnd.
      * cbn [hm_params_kinds make_handler_metadata].
        apply kept_kinds_cc.
      * reflexivity.
      * reflexivity.
Qed.

Lemma pdescs_of_firstn (m : nat) :
  ∀ (ds : list decision) (keeps : list bool) (kinds : list flambda_kind),
    pdescs_of (firstn m ds) (firstn m keeps) (firstn m kinds)
    = firstn m (pdescs_of ds keeps kinds).
Proof.
  induction m as [ | m IH]; intros ds keeps kinds; [ reflexivity | ].
  destruct ds as [ | d ds], keeps as [ | b keeps], kinds as [ | k kinds];
    reflexivity || (cbn [firstn pdescs_of]; rewrite IH; reflexivity).
Qed.

Lemma pdescs_of_skipn (m : nat) :
  ∀ (ds : list decision) (keeps : list bool) (kinds : list flambda_kind),
    pdescs_of (skipn m ds) (skipn m keeps) (skipn m kinds)
    = skipn m (pdescs_of ds keeps kinds).
Proof.
  induction m as [ | m IH]; intros ds keeps kinds; [ reflexivity | ].
  destruct ds as [ | d ds], keeps as [ | b keeps], kinds as [ | k kinds];
    cbn [skipn pdescs_of]; try reflexivity;
    try (destruct (skipn m _); reflexivity);
    try (destruct (skipn m _), (skipn m _); reflexivity).
  apply IH.
Qed.

Lemma forallb_firstn_skipn {A : Type} (p : A -> bool) (l : list A) (m : nat) :
  forallb p l = true ->
  forallb p (firstn m l) = true ∧ forallb p (skipn m l) = true.
Proof.
  intros H. rewrite <- (firstn_skipn m l), forallb_app in H.
  apply andb_true_iff in H. exact H.
Qed.

Lemma forall2_firstn_skipn {A B : Type} (R : A -> B -> Prop) (m : nat) :
  ∀ (l : list A) (l' : list B),
    Forall2 R l l' ->
    Forall2 R (firstn m l) (firstn m l') ∧ Forall2 R (skipn m l) (skipn m l').
Proof.
  induction m as [ | m IH]; intros l l' H;
    [ split; [ constructor | exact H ] | ].
  destruct H as [ | a b l l' Hab H]; [ split; constructor | ].
  destruct (IH l l' H) as [H1 H2].
  split; [ constructor; assumption | exact H2 ].
Qed.

Lemma args_ok2_app (kenv : finite_map flambda_kind) (ps1 : list pdesc) :
  ∀ (ps2 : list pdesc) (a1 a2 : list simple),
    length a1 = length ps1 ->
    args_ok2 sol unboxed kenv (ps1 ++ ps2) (a1 ++ a2)
    = args_ok2 sol unboxed kenv ps1 a1 && args_ok2 sol unboxed kenv ps2 a2.
Proof.
  induction ps1 as [ | p ps1 IH]; intros ps2 [ | a a1] a2 Hl;
    simpl in Hl; try discriminate Hl; [ reflexivity | ].
  cbn [app args_ok2]. rewrite IH by lia. apply andb_assoc.
Qed.

Lemma args_ok2_split (kenv : finite_map flambda_kind) (m : nat)
    (ps : list pdesc) (args : list simple) :
  length args = length ps ->
  args_ok2 sol unboxed kenv ps args = true ->
  args_ok2 sol unboxed kenv (firstn m ps) (firstn m args) = true
  ∧ args_ok2 sol unboxed kenv (skipn m ps) (skipn m args) = true.
Proof.
  intros Hl H. rewrite <- (firstn_skipn m ps), <- (firstn_skipn m args),
    args_ok2_app in H by (rewrite !length_firstn; lia).
  apply andb_true_iff in H. exact H.
Qed.

Lemma args_rel2_app (sh1 sh2 : list pshape) (a1 a2 b1 b2 : list simple) :
  args_rel2 unboxed dead sh1 a1 b1 -> length a1 = length sh1 ->
  args_rel2 unboxed dead sh2 a2 b2 ->
  args_rel2 unboxed dead (sh1 ++ sh2) (a1 ++ a2) (b1 ++ b2).
Proof.
  induction 1 as [args Hargs | o sh | sh a args args' Ha _ IH
                 | sh kind a a' args args' Ha _ IH
                 | sh fields fields_y y pairs args args' Hy Hpairs _ IH];
    intros Hl H2.
  - destruct args; [ exact H2 | discriminate Hl ].
  - discriminate Hl.
  - cbn [app]. apply args_rel2_plain; [ exact Ha | ].
    apply IH; [ simpl in Hl; lia | exact H2 ].
  - cbn [app]. apply args_rel2_dead; [ exact Ha | ].
    apply IH; [ simpl in Hl; lia | exact H2 ].
  - cbn [app]. rewrite <- app_assoc.
    apply (args_rel2_unboxed unboxed dead _ fields fields_y y pairs);
      [ exact Hy | exact Hpairs | ].
    apply IH; [ simpl in Hl; lia | exact H2 ].
Qed.

Lemma shape_kinds_app (sh1 : list pshape) :
  ∀ (sh2 : list pshape) (k1 k2 : list flambda_kind),
    length k1 = length sh1 ->
    shape_kinds (sh1 ++ sh2) (k1 ++ k2)
    = shape_kinds sh1 k1 ++ shape_kinds sh2 k2.
Proof.
  induction sh1 as [ | o sh1 IH]; intros sh2 [ | k k1] k2 Hl;
    simpl in Hl; try discriminate Hl; [ reflexivity | ].
  destruct o; cbn [app shape_kinds]; rewrite IH by lia;
    rewrite ?app_assoc; reflexivity.
Qed.

Lemma get_args_count (en : rebuild_env) (args : list simple) :
  ∀ (pd : list decision) (pd_cc : list param_decision)
    (kinds : list flambda_kind),
    length args = length pd_cc -> length pd = length pd_cc ->
    length kinds = length pd_cc ->
    length (get_args_with_kinds sol en pd_cc args)
    = length (flat_map pshape_of (pdescs_of pd (map is_kept pd_cc) kinds)).
Proof.
  induction args as [ | a args IH]; intros [ | d pd] [ | dcc pd_cc]
    [ | k kinds] H1 H2 H3; simpl in H1, H2, H3; try discriminate H1;
    try discriminate H2; try discriminate H3; [ reflexivity | ].
  unfold get_args_with_kinds in *. cbn [combine flat_map map pdescs_of].
  rewrite !length_app, (IH pd pd_cc kinds) by lia.
  unfold pshape_of. cbn [pd_keep]. destruct dcc; reflexivity.
Qed.

(* The arguments of a call of code that changes its calling convention,
   group by group. *)
Lemma call_args_groups (en : rebuild_env) (enu : unbox_env) :
  env_kinds (ue_env enu) = env_kinds en ->
  (∀ x, kind_ok (kind_of_variable en x) = true) ->
  ∀ (sizes : list nat) (args : list simple) (pd : list decision)
    (pd_cc : list param_decision) (kinds : list flambda_kind),
    (∀ z, z ∈ variables_in_simples args -> fresh z) ->
    Forall2 code_link pd pd_cc ->
    length args = length kinds -> length pd = length kinds ->
    length pd_cc = length kinds -> list_sum sizes = length kinds ->
    forallb pdesc_ok (pdescs_of pd (map is_kept pd_cc) kinds) = true ->
    args_ok2 sol unboxed (env_kinds en)
      (pdescs_of pd (map is_kept pd_cc) kinds) args
    = true ->
    ∃ groups : list (list (simple * flambda_kind)),
      all_some
        (map
           (fun '(args, decisions) =>
              get_args_with_kinds_u sol unboxed (ue_env enu) decisions args)
           (combine (group_by_sizes sizes args) (group_by_sizes sizes pd)))
      = Some groups
      ∧ args_rel2 unboxed dead
          (flat_map pshape_of (pdescs_of pd (map is_kept pd_cc) kinds))
          (flat_map (map fst)
             (map
                (fun '(args, decisions) =>
                   get_args_with_kinds sol en decisions args)
                (combine (group_by_sizes sizes args)
                   (group_by_sizes sizes pd_cc))))
          (flat_map (map fst) groups)
      ∧ map (map snd) groups = map decisions_kinds (group_by_sizes sizes pd)
      ∧ shape_kinds
          (flat_map pshape_of (pdescs_of pd (map is_kept pd_cc) kinds))
          (flat_map (map snd)
             (map
                (fun '(args, decisions) =>
                   get_args_with_kinds sol en decisions args)
                (combine (group_by_sizes sizes args)
                   (group_by_sizes sizes pd_cc))))
        = decisions_kinds pd.
Proof.
  intros Hkinds Hkok. induction sizes as [ | m sizes IH];
    intros args pd pd_cc kinds Hfresh Hlink Hla Hlp Hlc Hsum Hok Hargs.
  - cbn [list_sum] in Hsum.
    destruct kinds; [ | discriminate Hsum ].
    destruct args; [ | discriminate Hla ].
    destruct pd; [ | discriminate Hlp ].
    destruct pd_cc; [ | discriminate Hlc ].
    exists []. split; [ reflexivity | ].
    split; [ apply args_rel2_beyond; reflexivity | ].
    split; reflexivity.
  - change (list_sum (m :: sizes)) with (m + list_sum sizes) in Hsum.
    set (keeps := map is_kept pd_cc) in *.
    assert (Hpd_split : pdescs_of pd keeps kinds
                        = pdescs_of (firstn m pd) (firstn m keeps)
                            (firstn m kinds)
                          ++ pdescs_of (skipn m pd) (skipn m keeps)
                               (skipn m kinds))
      by (rewrite pdescs_of_firstn, pdescs_of_skipn;
          symmetry; apply firstn_skipn).
    assert (Hlk : length keeps = length kinds)
      by (unfold keeps; rewrite length_map; exact Hlc).
    assert (Hlen1 : length (pdescs_of pd keeps kinds) = length kinds)
      by (rewrite pdescs_of_length; lia).
    destruct (forall2_firstn_skipn code_link m pd pd_cc Hlink)
      as [Hlink1 Hlink2].
    destruct (forallb_firstn_skipn pdesc_ok _ m Hok) as [Hok1 Hok2].
    rewrite <- pdescs_of_firstn in Hok1. rewrite <- pdescs_of_skipn in Hok2.
    destruct (args_ok2_split (env_kinds en) m _ args
                ltac:(rewrite Hlen1; exact Hla) Hargs)
      as [Hargs1 Hargs2].
    rewrite <- pdescs_of_firstn in Hargs1.
    rewrite <- pdescs_of_skipn in Hargs2.
    unfold keeps in Hok1, Hok2, Hargs1, Hargs2.
    rewrite firstn_map in Hok1, Hargs1.
    rewrite skipn_map in Hok2, Hargs2.
    assert (Hfresh1 : ∀ z, z ∈ variables_in_simples (firstn m args) ->
                           fresh z).
    { intros z Hz. apply Hfresh. unfold variables_in_simples in *.
      rewrite <- (firstn_skipn m args), flat_map_app.
      apply in_or_app. left. exact Hz. }
    assert (Hfresh2 : ∀ z, z ∈ variables_in_simples (skipn m args) ->
                           fresh z).
    { intros z Hz. apply Hfresh. unfold variables_in_simples in *.
      rewrite <- (firstn_skipn m args), flat_map_app.
      apply in_or_app. right. exact Hz. }
    destruct (call_args_rel en enu (firstn m args) Hkinds Hkok Hfresh1
                (firstn m pd) (firstn m pd_cc) (firstn m kinds) Hlink1)
      as (g & Hg & Hrel1 & Hk1 & Hk2);
      [ rewrite pdescs_of_length, !length_map, !length_firstn; lia
      | rewrite !length_firstn; lia | rewrite !length_firstn; lia
      | exact Hok1 | exact Hargs1 | ].
    destruct (IH (skipn m args) (skipn m pd) (skipn m pd_cc) (skipn m kinds)
                Hfresh2 Hlink2)
      as (gs & Hgs & Hrel2 & Hk3 & Hk4);
      [ rewrite !length_skipn; lia | rewrite !length_skipn; lia
      | rewrite !length_skipn; lia | rewrite length_skipn; lia
      | exact Hok2 | exact Hargs2 | ].
    cbn [group_by_sizes combine map all_some fold_right].
    fold (all_some
            (map
               (fun '(args, decisions) =>
                  get_args_with_kinds_u sol unboxed (ue_env enu) decisions
                    args)
               (combine (group_by_sizes sizes (skipn m args))
                  (group_by_sizes sizes (skipn m pd))))).
    rewrite Hg, Hgs.
    exists (g :: gs). split; [ reflexivity | ].
    assert (Hcount : length (get_args_with_kinds sol en (firstn m pd_cc)
                               (firstn m args))
                     = length (flat_map pshape_of
                                 (pdescs_of (firstn m pd)
                                    (map is_kept (firstn m pd_cc))
                                    (firstn m kinds))))
      by (apply get_args_count; rewrite !length_firstn; lia).
    rewrite Hpd_split. unfold keeps. rewrite firstn_map, skipn_map.
    cbn [flat_map map]. rewrite flat_map_app.
    split; [ | split ].
    + apply args_rel2_app; [ exact Hrel1 | | exact Hrel2 ].
      rewrite length_map. exact Hcount.
    + cbn [map]. rewrite Hk1, Hk3. reflexivity.
    + rewrite shape_kinds_app by (rewrite length_map; exact Hcount).
      rewrite Hk2, Hk4. unfold decisions_kinds.
      rewrite <- flat_map_app, firstn_skipn. reflexivity.
Qed.

Lemma list_sum_unarize (a : arity) :
  list_sum (map (@length flambda_kind) (unarize_per_parameter a))
  = length (unarize a).
Proof.
  unfold unarize_per_parameter, unarize.
  induction a as [ | c a IH]; [ reflexivity | ].
  cbn [map flat_map].
  transitivity (length (unarize_component c)
                + list_sum (map (@length _) (map unarize_component a)));
    [ reflexivity | ].
  rewrite IH, length_app. reflexivity.
Qed.

Lemma prepend_nil {A : Type} (gs : list (list A)) : prepend_first [] gs = gs.
Proof. destruct gs; reflexivity. Qed.

Lemma unarize_arity_of_groups (gs : list (list (simple * flambda_kind))) :
  unarize (arity_of_groups gs) = flat_map (map snd) gs.
Proof.
  induction gs as [ | g gs IH]; [ reflexivity | ].
  unfold arity_of_groups in *. cbn [map flat_map]. unfold unarize in *.
  cbn [flat_map]. rewrite IH. f_equal. cbn [unarize_component].
  induction g as [ | [s k] g IHg]; [ reflexivity | ].
  cbn [map flat_map unarize_component app]. rewrite IHg. reflexivity.
Qed.

Lemma unarize_grouped_u (gs : list (list decision)) :
  unarize (arity_of_grouped_decisions_u gs) = flat_map decisions_kinds gs.
Proof.
  induction gs as [ | g gs IH]; [ reflexivity | ].
  unfold arity_of_grouped_decisions_u, unarize in *. cbn [map flat_map].
  rewrite IH. f_equal. cbn [unarize_component].
  induction (decisions_kinds g) as [ | k ks IHk]; [ reflexivity | ].
  cbn [map flat_map unarize_component app]. rewrite IHk. reflexivity.
Qed.

Lemma decisions_kinds_prepend (gs : list (list decision)) :
  map decisions_kinds (prepend_first [Decided Delete] gs)
  = map decisions_kinds gs.
Proof. destruct gs; reflexivity. Qed.

Lemma arity_groups_eq (gs : list (list (simple * flambda_kind)))
    (gpd : list (list decision)) :
  map (map snd) gs = map decisions_kinds gpd ->
  arity_of_groups gs
  = arity_of_grouped_decisions_u (prepend_first [Decided Delete] gpd).
Proof.
  intros H. unfold arity_of_groups, arity_of_grouped_decisions_u.
  rewrite <- (map_map decisions_kinds
                (fun ks => Unboxed_product (map Singleton ks))).
  rewrite decisions_kinds_prepend, <- H, map_map.
  apply map_ext. intros g. f_equal. apply singletons_snd.
Qed.

(* The decisions of the results of code that changes its calling
   convention, in each run. *)
Lemma rds_decisions (nodes : list node) (kinds : list flambda_kind) :
  length (pdescs_of (decisions_u sol unboxed nodes kinds)
            (map is_kept (decisions_of sol nodes kinds)) kinds)
  = length kinds ->
  let rds := pdescs_of (decisions_u sol unboxed nodes kinds)
               (map is_kept (decisions_of sol nodes kinds)) kinds in
  map cc_dec rds = decisions_of sol nodes kinds
  ∧ map pd_kind rds = kinds
  ∧ map pd_decision rds = decisions_u sol unboxed nodes kinds.
Proof.
  unfold decisions_u, decisions_of. revert kinds.
  induction nodes as [ | n nodes IH]; intros [ | k kinds] Hl;
    try discriminate Hl; cbv zeta;
    [ split; [ | split ]; reflexivity | | ].
  - cbn. split; [ | split ]; reflexivity.
  - cbn [combine map pdescs_of] in Hl |- *. simpl in Hl.
    injection Hl as Hl. destruct (IH kinds Hl) as (H1 & H2 & H3).
    cbn [map]. split; [ | split ]; f_equal; try assumption.
    unfold cc_dec. cbn [pd_keep pd_kind].
    destruct (is_node_used sol n k); reflexivity.
Qed.

Lemma flat_map_prepend_delete (gs : list (list decision)) :
  flat_map decisions_kinds (prepend_first [Decided Delete] gs)
  = flat_map decisions_kinds gs.
Proof. destruct gs; reflexivity. Qed.

Lemma flat_map_decisions_concat (gs : list (list decision)) :
  flat_map decisions_kinds gs = decisions_kinds (concat gs).
Proof.
  induction gs as [ | g gs IH]; [ reflexivity | ].
  cbn [flat_map concat]. rewrite IH. unfold decisions_kinds.
  rewrite flat_map_app. reflexivity.
Qed.

Lemma wrapper_args_keep (mask : list bool) :
  ∀ (kinds : list flambda_kind) (n : positive),
    length mask = length kinds ->
    wrapper_args n mask (map (fun k => (Keep k, k)) kinds)
    = filter_kept mask
        (map Var (FlambdaFreshQuantifierSpec.names_from (length kinds) n)).
Proof.
  induction mask as [ | b mask IH]; intros [ | k kinds] n Hl;
    simpl in Hl; try discriminate Hl; [ reflexivity | ].
  cbn [map wrapper_args length FlambdaFreshQuantifierSpec.names_from
       filter_kept].
  rewrite IH by lia. destruct b; reflexivity.
Qed.

(* The wrapper of a call of code that keeps its calling convention, in the
   first run, is that of the change of calling conventions for a callee
   that keeps every result. *)
Lemma make_apply_wrapper_as_cc (en : rebuild_env) (ap : apply_expr) :
  (∃ mask : list bool,
     lookup (ap_continuation ap) (env_conts en) = Some mask
     ∧ length mask = length (unarize (ap_return_arity ap)))
  ∨ lookup (ap_continuation ap) (env_conts en) = None ->
  make_apply_wrapper ff en ap
  = make_apply_wrapper_cc ff en ap
      (map (fun k => (Keep k, k)) (unarize (ap_return_arity ap))).
Proof.
  intros Hm. unfold make_apply_wrapper, make_apply_wrapper_cc,
    apply_decisions.
  rewrite length_map.
  set (kinds := unarize (ap_return_arity ap)) in *.
  destruct Hm as [(mask & Hm & Hl) | Hm]; rewrite Hm.
  - rewrite firstn_app, Hl, Nat.sub_diag, firstn_O, app_nil_r, <- Hl,
      firstn_all.
    assert (Hf : forallb
                   (fun '(keep, (decision, _)) =>
                      Bool.eqb keep (is_kept decision))
                   (combine mask (map (fun k => (Keep k, k)) kinds))
                 = forallb (fun keep => keep) mask).
    { clearbody kinds. clear -Hl. revert kinds Hl.
      induction mask as [ | b mask IH]; intros [ | k kinds] Hl;
        simpl in Hl; try discriminate Hl; [ reflexivity | ].
      cbn [map combine forallb is_kept]. rewrite (IH kinds ltac:(lia)).
      destruct b; reflexivity. }
    rewrite Hf. destruct (forallb (fun keep => keep) mask); [ reflexivity | ].
    assert (Hkk : kept_kinds (map fst (map (fun k => (Keep k, k)) kinds))
                  = kinds).
    { clearbody kinds. clear. induction kinds as [ | k kinds IH];
        [ reflexivity | ].
      unfold kept_kinds in *. cbn [map flat_map app]. rewrite IH.
      reflexivity. }
    rewrite Hkk, (wrapper_args_keep mask kinds ff Hl), Hl.
    reflexivity.
  - cbn [app]. rewrite firstn_all2 by (rewrite repeat_length; lia).
    assert (Hf : ∀ l : list flambda_kind,
              forallb
                (fun '(keep, (decision, _)) =>
                   Bool.eqb keep (is_kept decision))
                (combine (repeat true (length l))
                   (map (fun k => (Keep k, k)) l))
              = true).
    { induction l as [ | k l IH]; [ reflexivity | exact IH ]. }
    rewrite Hf. reflexivity.
Qed.

(* The results of code that keeps its calling convention, kept by the
   callee, with a use or not. *)
Lemma nc_rds (nodes : list node) (kinds : list flambda_kind) :
  existsb is_unbox (fixed_arity_decisions sol unboxed nodes kinds) = false ->
  length
    (pdescs_of (fixed_arity_decisions sol unboxed nodes kinds)
       (map (fun _ => true) (fixed_arity_decisions sol unboxed nodes kinds))
       kinds)
  = length kinds ->
  let rds :=
    pdescs_of (fixed_arity_decisions sol unboxed nodes kinds)
      (map (fun _ => true) (fixed_arity_decisions sol unboxed nodes kinds))
      kinds
  in
  combine (map cc_dec rds) (map pd_kind rds)
  = map (fun k => (Keep k, k)) kinds
  ∧ ∀ n : positive,
      freshen_decisions (map (fun k => (Decided (Keep k), k)) kinds) n
      = freshen_decisions (combine (map pd_decision rds) (map pd_kind rds))
          n.
Proof.
  unfold fixed_arity_decisions. revert kinds.
  induction nodes as [ | nd nodes IH]; intros [ | k kinds] Hu Hl;
    try discriminate Hl; cbv zeta.
  - split; [ reflexivity | intros n; reflexivity ].
  - split; [ reflexivity | intros n; reflexivity ].
  - cbn [combine map pdescs_of existsb] in Hu, Hl |- *.
    simpl in Hl. injection Hl as Hl.
    apply orb_false_iff in Hu as [Hu1 Hu].
    destruct (IH kinds Hu Hl) as [H1 H2]. cbv zeta in H1, H2.
    destruct (unboxed_of_node unboxed nd) as [fields | ];
      [ discriminate Hu1 | ].
    split.
    + cbn [map combine]. rewrite H1. reflexivity.
    + intros n. cbn [map combine].
      destruct (is_node_used sol nd k); cbn [freshen_decisions];
        rewrite H2; reflexivity.
Qed.

(* A call of code that keeps its calling convention, as the two rebuilds
   make it: the first by FlambdaReaperRebuild.rebuild_apply, the second as
   a call that gives the callee its arguments and expects its results as
   they are, with a wrapper for the results that the continuation does
   not take as they are. *)
Lemma call_rebuilt_nc (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  cinv C en enu ->
  call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed nr reach
    ovr rem (env_kinds en) C ap
  = true ->
  (∀ z, z ∈ variables_in_apply ap -> fresh z) ->
  ap_exn_continuation ap <> ff -> ap_exn_continuation ap <> ffu ->
  code_id_actually_called F sol ap <> None ->
  apply_change_u F sol cannot_change code_deps unit_code_ids codes unboxed ap
  = Not_changing_u ->
  uc_result (K_of C) (live_conts F sol code_deps codes (Apply ap))
    (cc_rebuild en (Apply ap)) (u_rebuild enu (Apply ap)).
Proof.
  intros Hci Hok Hfresh Hff Hffu Hsome Hac.
  change (live_conts F sol code_deps codes (Apply ap))
    with (if apply_never_returns F sol code_deps codes ap
          then [ap_exn_continuation ap]
          else free_continuations_of_expr (Apply ap)).
  destruct ap as [callee ckind args k exn extra AA RA].
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity] in *.
  set (ap := {| ap_callee := callee; ap_call_kind := ckind; ap_args := args;
                ap_continuation := k; ap_exn_continuation := exn;
                ap_exn_extra_args := extra; ap_args_arity := AA;
                ap_return_arity := RA |}) in *.
  unfold call_ok_f3 in Hok. cbn [ap_call_kind ap_callee ap ap_args
    ap_continuation ap_exn_continuation ap_exn_extra_args ap_args_arity
    ap_return_arity] in Hok.
  assert (Hck : ckind <> Indirect_unknown_arity)
    by (intros ->; apply Hsome; reflexivity).
  destruct callee as [f | sym | cst]; try discriminate Hok.
  destruct (code_id_actually_called F sol ap) as [id | ] eqn:Hcalled;
    [ | exfalso; exact (Hsome eq_refl) ].
  apply andb_true_iff in Hok as [Hok Hrest].
  apply andb_true_iff in Hok as [Hok Hnr].
  apply andb_true_iff in Hok as [Hok Hfused].
  apply andb_true_iff in Hok as [_ Hfplain].
  destruct (lookup id codes) as [md | ] eqn:Hcodes; [ | discriminate Hrest ].
  assert (Hnc : code_change_u sol cannot_change code_deps unit_code_ids
                  unboxed id md
                = Not_changing_u).
  { unfold apply_change_u in Hac. rewrite Hcalled, Hcodes in Hac.
    exact Hac. }
  destruct (code_pdescs sol cannot_change code_deps unit_code_ids unboxed id
              md)
    as [[pds rds] | ] eqn:Hpd; [ | discriminate Hrest ].
  apply andb_true_iff in Hrest as [Hrest Hret].
  apply andb_true_iff in Hrest as [Hrest Hkept].
  rewrite Hnc in Hkept.
  apply andb_true_iff in Hrest as [Hrest Hexn].
  apply andb_true_iff in Hrest as [Hrest Hargs].
  apply andb_true_iff in Hrest as [Hrest Hrok].
  apply andb_true_iff in Hrest as [Hrest Hdok].
  apply andb_true_iff in Hrest as [Hrest Hrlen].
  apply andb_true_iff in Hrest as [Hrest Hdlen].
  apply andb_true_iff in Hrest as [Hrest Halen].
  apply andb_true_iff in Hrest as [Hrest Hmcd].
  apply andb_true_iff in Hrest as [Haa Hra].
  apply arity_eqb_eq in Haa, Hra. subst AA RA.
  apply Nat.eqb_eq in Halen, Hdlen, Hrlen.
  destruct (code_exn_decision sol code_deps unboxed id) as [de | ] eqn:Hde;
    [ | discriminate Hexn ].
  destruct (code_exn_decision_cases id de Hde) as (Hdc & _).
  destruct (exn_rebuilt en enu C ap de Hci Hdc Hexn)
    as (extra_u & Hxu & Hexrel).
  { intros z Hz. apply Hfresh. unfold variables_in_apply.
    apply in_or_app. right. apply in_or_app. right. exact Hz. }
  assert (Hncc : code_calling_convention_change sol cannot_change code_deps
                   unit_code_ids id md
                 = Not_changing_calling_convention).
  { unfold code_change_u in Hnc. unfold code_calling_convention_change.
    destruct (cannot_change_calling_convention cannot_change unit_code_ids
                id);
      [ reflexivity | ].
    destruct (lookup id code_deps); [ discriminate Hnc | reflexivity ]. }
  pose proof Hpd as Hpd'.
  unfold code_pdescs in Hpd. rewrite Hnc, Hncc in Hpd.
  destruct (lookup id code_deps) as [cd | ] eqn:Hcd; [ | discriminate Hpd ].
  assert (Hmask_u : kept_code_mask F sol code_deps codes ap
                    = Some (map (s_has_usage sol) (cd_params cd)))
    by (unfold kept_code_mask; rewrite Hcalled, Hcodes, Hcd; reflexivity).
  assert (Hnr_ap : apply_never_returns F sol code_deps codes ap
                   = code_never_returns sol code_deps id)
    by (unfold apply_never_returns; rewrite Hcalled, Hcodes; reflexivity).
  destruct (cm_is_tupled md
            || existsb is_unbox
                 (fixed_arity_decisions sol unboxed (cd_return cd)
                    (unarize (cm_result_arity md))))
    eqn:Htup; [ discriminate Hpd | ].
  apply orb_false_iff in Htup as [_ Hnu].
  injection Hpd as Hpds Hrds. subst pds rds.
  set (kinds := unarize (cm_params_arity md)) in *.
  set (kinds_r := unarize (cm_result_arity md)) in *.
  set (rd := fixed_arity_decisions sol unboxed (cd_return cd) kinds_r) in *.
  set (rds := pdescs_of rd (map (fun _ => true) rd) kinds_r) in *.
  assert (Hfx : fields_of unboxed f = None).
  { cbn in Hfplain. apply negb_true_iff in Hfplain.
    unfold Keyed.is_bound in Hfplain. unfold fields_of.
    destruct (Keyed.lookup _ _ _); [ discriminate Hfplain | reflexivity ]. }
  assert (Hdead_eq :
    simple_is_dead sol (ue_env enu) (Var f)
    || existsb (simple_is_dead sol (ue_env enu)) args
    = simple_is_dead sol en (Var f) || existsb (simple_is_dead sol en) args).
  { f_equal; [ apply simple_is_dead_kinds, (ci_kinds _ _ _ Hci) | ].
    clear -Hci. induction args as [ | a args IH]; [ reflexivity | ].
    simpl. rewrite IH, (simple_is_dead_kinds _ _ a (ci_kinds _ _ _ Hci)).
    reflexivity. }
  unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
  unfold rebuild_apply_cc, apply_calling_convention_change.
  rewrite Hcalled. unfold get_calling_convention_change.
  rewrite Hcodes, Hncc.
  unfold rebuild_apply, rebuild_apply_u. rewrite Hac. unfold ap in *.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  rewrite Hdead_eq.
  destruct (simple_is_dead sol en (Var f) || existsb (simple_is_dead sol en)
              args) eqn:Hdead;
    [ exact (uc_result_invalid _ _) | ].
  change (unboxed_of unboxed (Var f)) with (fields_of unboxed f).
  rewrite Hfx.
  rewrite (rewrite_call_kind_direct
             {| ap_callee := Var f; ap_call_kind := ckind;
                ap_args := args; ap_continuation := k;
                ap_exn_continuation := exn; ap_exn_extra_args := extra;
                ap_args_arity := cm_params_arity md;
                ap_return_arity := cm_result_arity md |}
             id Hcalled).
  rewrite Hxu. cbn beta iota zeta.
  (* The arguments, the same in both runs. *)
  rewrite Hnr_ap. unfold kept_arguments_u. cbn beta iota.
  rewrite Hmask_u. cbn beta iota.
  unfold kept_args_ok in Hkept. rewrite Hmask_u in Hkept.
  unfold rebuild_arguments.
  cbn [ap_callee ap_call_kind ap_args node_of_simple].
  rewrite !(known_kind_match ckind _ _ Hck).
  unfold masked_args_ok in Hkept. cbn [ap_args] in Hkept.
  apply andb_true_iff in Hkept as [Hmeq Hmused].
  apply forallb2_eqb_eq in Hmeq. rewrite <- Hmeq.
  rewrite (mask_arguments_kinds (ue_env enu) en args (ci_kinds _ _ _ Hci)).
  set (m := arguments_used_by_known_arity_call F sol (Node_variable f) args)
    in *.
  set (A := mask_arguments en m args).
  rewrite length_map in Halen.
  assert (Hfargs : ∀ z, z ∈ variables_in_simples args -> fresh z).
  { intros z Hz. apply Hfresh. unfold variables_in_apply.
    cbn [ap_callee ap_args variables_in_simple app]. right.
    apply in_or_app. left. exact Hz. }
  set (cs := Code_shape
               (flat_map pshape_of
                  (map (fun kind => Pdesc (Decided (Keep kind)) true kind)
                     kinds))
               (flat_map pshape_of rds) (flat_map kept_kind rds)
               (cm_params_arity md) (cm_result_arity md)
               (pshape1 (Pdesc de true Value)) true).
  assert (HCS : CSf id = Some cs).
  { unfold CSf. rewrite Hcodes, Hde, Hnc, Hpd'. reflexivity. }
  match goal with
  | |- uc_result _ _ (make_apply_wrapper _ _ ?X)
         (make_apply_wrapper_u _ _ _ ?Y _ _) =>
    set (ap1 := X); set (ap1u := Y)
  end.
  assert (Hcall : call_rel unboxed dead CSf RCHf OVf REMf ap1 ap1u cs).
  { split;
      [ left; exists id;
        split; [ reflexivity | split; [ reflexivity | exact HCS ] ] | ].
    split; [ reflexivity | ].
    split; [ apply (plain2_of f Value Hfx); [ | reflexivity | exact Hfused ];
             apply Hfresh; left; reflexivity | ].
    split; [ cbn [cs cs_params ap1 ap1u ap_args]; rewrite plain_shapes;
             exact (args_rel2_mask en args m kinds Halen Hmused Hfargs) | ].
    split; [ cbn [cs cs_params]; rewrite plain_shapes;
             apply plain_shapes_no_dead | ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | ].
    split; [ cbn [cs cs_params ap1 ap_args_arity cs_params_arity];
             rewrite plain_shapes, plain_shapes_kinds;
             reflexivity | ].
    reflexivity. }
  assert (Hexn_ok : exn_call_rel unboxed dead (K_of C) (cs_exn cs) ap1 ap1u)
    by exact (Hexrel ap1 ap1u eq_refl eq_refl eq_refl).
  destruct (code_never_returns sol code_deps id) eqn:Hnrc.
  { (* A call of code that never returns. *)
    cbn [negb orb] in Hnr.
    refine (uc_result_never (K_of C) _ ap1 ap1u cs id _ _ _ Hcall Hexn_ok
              eq_refl HCS Hnr _ _ (make_apply_wrapper_cases en ap1) _).
    - intros H. exact (Hff (eq_sym H)).
    - intros H. exact (Hffu (eq_sym H)).
    - intros x Hx. exact Hx. }
  cbn [orb] in Hret.
  destruct (lookup k C) as [[kps [ | ]] | ] eqn:HCk; try discriminate Hret.
  assert (HKk : lookup k (K_of C)
                = Some (Kshape (flat_map pshape_of kps)
                          (flat_map kept_kind kps) true))
    by (rewrite lookup_K_of, HCk; reflexivity).
  pose proof (ci_ok _ _ _ Hci k kps true HCk) as Hkps_ok.
  assert (Hlen_kr : length kps = length rds)
    by exact (forallb2_length _ _ _ Hret).
  destruct (apply_decisions_kps C en enu k kps true Hci HCk)
    as [Happ Hmask].
  rewrite Hlen_kr in Happ.
  destruct (nc_rds (cd_return cd) kinds_r Hnu Hrlen) as [Hcomb Hfresh_eq].
  fold rd rds in Hcomb, Hfresh_eq.
  set (ap0 := Apply_expr (Var f) ckind args k exn extra
                (cm_params_arity md) (cm_result_arity md)).
  rewrite (make_apply_wrapper_as_cc en ap1).
  2: { destruct Hmask as [Hm | [Hm _]];
       [ left; exists (map pd_keep kps);
         split; [ exact Hm | rewrite length_map, Hlen_kr; exact Hrlen ]
       | right; exact Hm ]. }
  change (unarize (ap_return_arity ap1)) with kinds_r. rewrite <- Hcomb.
  apply (uc_result_results (K_of C) ap0 ap1 ap1u cs k kps rds en enu);
    try reflexivity; try assumption.
  - intros H. exact (Hff (eq_sym H)).
  - intros H. exact (Hffu (eq_sym H)).
  - rewrite (ci_dec _ _ _ Hci k), HCk. reflexivity.
  - apply Hfresh_eq.
Qed.
(* The arguments of a call of a variable, at the mask of the call. *)
Lemma rebuild_arguments_mask (s : solution) (en : rebuild_env)
    (ap : apply_expr) (f : variable) :
  ap_callee ap = Var f ->
  rebuild_arguments F s en ap
  = mask_arguments en (call_mask F s f ap) (ap_args ap).
Proof.
  intros Hc. unfold rebuild_arguments, call_mask. rewrite Hc.
  cbn [node_of_simple]. destruct (ap_call_kind ap); reflexivity.
Qed.

(* The results of code that keeps its calling convention, as the code
   dep gives them: kept, with a use or not. *)
Definition kept_result (r : pdesc) : Prop :=
  pd_keep r = true
  ∧ (pd_decision r = Decided (Keep (pd_kind r))
     ∨ pd_decision r = Keep_unused (pd_kind r)).

Lemma fixed_rds_kept (nodes : list node) (kinds : list flambda_kind) :
  existsb is_unbox (fixed_arity_decisions sol unboxed nodes kinds) = false ->
  Forall kept_result
    (pdescs_of (fixed_arity_decisions sol unboxed nodes kinds)
       (map (fun _ => true) (fixed_arity_decisions sol unboxed nodes kinds))
       kinds).
Proof.
  unfold fixed_arity_decisions. revert kinds.
  induction nodes as [ | nd nodes IH]; intros [ | k kinds] Hu;
    [ constructor | constructor | constructor | ].
  cbn [existsb combine map] in Hu. apply orb_false_iff in Hu as [Hu1 Hu].
  cbn [combine map pdescs_of].
  constructor; [ | exact (IH kinds Hu) ].
  destruct (unboxed_of_node unboxed nd) as [fields | ];
    [ discriminate Hu1 | ].
  split; [ reflexivity | ].
  destruct (is_node_used sol nd k); [ left | right ]; reflexivity.
Qed.

Lemma kept_results_kinds (rds : list pdesc) :
  Forall kept_result rds -> flat_map kept_kind rds = map pd_kind rds.
Proof.
  induction 1 as [ | r rds [Hk _] _ IH]; [ reflexivity | ].
  cbn [flat_map map]. unfold kept_kind at 1. rewrite Hk, IH. reflexivity.
Qed.

Lemma pdescs_of_kinds (kinds : list flambda_kind) :
  ∀ (ds : list decision) (keeps : list bool),
    length (pdescs_of ds keeps kinds) = length kinds ->
    map pd_kind (pdescs_of ds keeps kinds) = kinds.
Proof.
  induction kinds as [ | k kinds IH]; intros [ | d ds] [ | b keeps] Hl;
    cbn [pdescs_of length] in Hl |- *; try discriminate Hl;
    try reflexivity.
  cbn [map pd_kind]. rewrite (IH ds keeps ltac:(lia)). reflexivity.
Qed.

Lemma ind_results_kinds (kps : list pdesc) :
  ∀ kinds : list flambda_kind,
    length kps = length kinds ->
    flat_map kept_kind (ind_results kps kinds) = kinds.
Proof.
  induction kps as [ | kp kps IH]; intros [ | k kinds] Hl; simpl in Hl;
    try discriminate Hl; [ reflexivity | ].
  cbn [ind_results flat_map]. unfold kept_kind at 1. cbn [pd_keep pd_kind].
  rewrite (IH kinds ltac:(lia)). reflexivity.
Qed.

Lemma ind_results_no_unboxed (kps : list pdesc) :
  ∀ kinds : list flambda_kind,
    no_unboxed (flat_map pshape_of (ind_results kps kinds)).
Proof.
  induction kps as [ | kp kps IH]; intros [ | k kinds] fields Hin;
    [ exact Hin | exact Hin | exact Hin | ].
  cbn [ind_results flat_map] in Hin. apply in_app_or in Hin as [Hin | Hin];
    [ | exact (IH kinds fields Hin) ].
  unfold pshape_of in Hin. cbn [pd_keep pd_decision] in Hin.
  destruct (pd_decision kp) as [[kk | ] | fs | kk];
    destruct Hin as [Hin | []]; discriminate Hin.
Qed.

Lemma ind_results_ok (kps : list pdesc) :
  ∀ kinds : list flambda_kind,
    forallb pdesc_ok kps = true ->
    forallb2 result_ok kps (ind_results kps kinds) = true ->
    forallb pdesc_ok (ind_results kps kinds) = true.
Proof.
  induction kps as [ | kp kps IH]; intros [ | k kinds] Hok Hres;
    try reflexivity.
  cbn [ind_results forallb2] in Hres. cbn [forallb] in Hok.
  cbn [ind_results forallb].
  apply andb_true_iff in Hok as [Hkp Hok].
  apply andb_true_iff in Hres as [Hr Hres].
  rewrite (IH kinds Hok Hres), andb_true_r.
  unfold result_ok in Hr. cbn [pd_kind] in Hr.
  apply andb_true_iff in Hr as [Hkind _]. apply kind_eqb_eq in Hkind.
  subst k.
  unfold pdesc_ok in Hkp |- *. apply andb_true_iff in Hkp as [Hkk _].
  cbn [pd_kind pd_decision pd_keep]. rewrite Hkk. cbn [andb].
  destruct (pd_decision kp) as [[kk | ] | fs | kk];
    cbn [pd_keep pd_kind andb]; apply kind_eqb_refl.
Qed.

Lemma ind_results_cc (kps : list pdesc) :
  ∀ kinds : list flambda_kind,
    length kps = length kinds ->
    combine (map cc_dec (ind_results kps kinds))
      (map pd_kind (ind_results kps kinds))
    = map (fun k => (Keep k, k)) kinds
    ∧ ∀ n : positive,
        freshen_decisions (map (fun k => (Decided (Keep k), k)) kinds) n
        = freshen_decisions
            (combine (map pd_decision (ind_results kps kinds))
               (map pd_kind (ind_results kps kinds)))
            n.
Proof.
  induction kps as [ | kp kps IH]; intros [ | k kinds] Hl; simpl in Hl;
    try discriminate Hl; [ split; [ reflexivity | intros n; reflexivity ] | ].
  destruct (IH kinds ltac:(lia)) as [H1 H2].
  cbn [ind_results map combine]. split.
  - cbn [cc_dec pd_keep pd_kind]. rewrite H1. reflexivity.
  - intros n. cbn [pd_decision pd_kind].
    destruct (pd_decision kp) as [[kk | ] | fs | kk]; cbn [freshen_decisions];
      rewrite H2; reflexivity.
Qed.

(* The results of code that a call whose code is not known may call fit
   those that the call takes. *)
Lemma ind_fits (kps : list pdesc) :
  ∀ (rds : list pdesc) (kinds : list flambda_kind),
    forallb2 result_ok kps rds = true ->
    forallb2 result_ok kps (ind_results kps kinds) = true ->
    Forall kept_result rds ->
    Forall2 fits (flat_map pshape_of rds)
      (flat_map pshape_of (ind_results kps kinds)).
Proof.
  induction kps as [ | kp kps IH]; intros [ | r rds] [ | k kinds] H1 H2 Hk;
    try discriminate H1; try discriminate H2; try exact (Forall2_nil _).
  cbn [forallb2 ind_results] in H1, H2.
  apply andb_true_iff in H1 as [Hr1 H1]. apply andb_true_iff in H2 as [Hr2 H2].
  inversion Hk as [ | r0 rds0 [Hkeep Hdec] Hk' ]; subst.
  cbn [flat_map]. apply Forall2_app; [ | exact (IH rds kinds H1 H2 Hk') ].
  unfold result_ok in Hr1, Hr2. cbn [pd_kind pd_decision] in Hr2.
  apply andb_true_iff in Hr1 as [Hkind1 Hr1].
  apply andb_true_iff in Hr2 as [Hkind2 _].
  apply kind_eqb_eq in Hkind1, Hkind2.
  unfold pshape_of. rewrite Hkeep. cbn [pd_keep pd_decision].
  destruct Hdec as [Hdec | Hdec]; rewrite Hdec in Hr1 |- *;
    destruct (pd_decision kp) as [[kk | ] | fs | kk];
    cbn in Hr1; try discriminate Hr1;
    constructor; try constructor.
  all: rewrite <- Hkind1, <- Hkind2; constructor.
Qed.

(* The shape at which a call whose code is not known is made: every code
   of the unit may be called at it. *)
(* A call of known arity whose code is not known is rebuilt as one. *)
Lemma rewrite_kind_known (F0 : facts) (s : solution) (pdc0 : bool)
    (ids : list code_id) (ap : apply_expr) :
  known_arity_call (ap_call_kind ap) = true ->
  code_id_actually_called F0 s ap = None ->
  known_arity_kind (rewrite_call_kind F0 s pdc0 ids ap) = true.
Proof.
  intros Hk Hc. unfold rewrite_call_kind.
  unfold code_id_actually_called in Hc.
  destruct (ap_call_kind ap) as [id | | cids] eqn:Hkind;
    try discriminate Hk.
  unfold called_call_kind.
  destruct (ap_callee ap) as [f | sym | cst]; [ | | reflexivity ];
    (destruct (code_id_actually_directly_called F0 s _) as [l | ];
     [ | reflexivity ];
     rewrite Hc;
     destruct (Keyed.is_empty l); [ rewrite Hkind | ]; reflexivity).
Qed.

Lemma known_arity_kind_call (kind : call_kind) :
  known_arity_kind kind = known_arity_call kind.
Proof. destruct kind; reflexivity. Qed.

Lemma ind_targets_k_gen (CS0 : code_id -> option code_shape)
    (kind kind' : call_kind) (cs : code_shape) :
  (known_arity_call kind = true -> known_arity_kind kind' = true) ->
  (∀ (id : code_id) (cs' : code_shape),
     CS0 id = Some cs' ->
     cs_keeps cs' = true
     ∧ Forall (fun o => o = Psh_plain) (cs_params cs')
     ∧ cs_result_kinds cs' = unarize (cs_result_arity cs')
     ∧ fits (cs_exn cs') (cs_exn cs)
     ∧ (cs_result_arity cs' = cs_result_arity cs ->
        Forall2 fits (cs_results cs') (cs_results cs))
     ∧ (known_arity_call kind = false ->
        cs_result_arity cs' = [Singleton Value] ->
        cs_results cs' = [Psh_plain])) ->
  Forall (fun o => o = Psh_plain) (cs_params cs) ->
  no_unboxed (cs_results cs) ->
  cs_result_kinds cs = unarize (cs_result_arity cs) ->
  ind_targets_k CS0 kind' cs.
Proof.
  intros Hkk Hcomp Hpl Hnu Hrk. unfold ind_targets_k.
  destruct (known_arity_kind kind') eqn:Hk'.
  - refine (conj _ (conj Hpl (conj Hnu Hrk))).
    intros id cs' Hcs'.
    destruct (Hcomp id cs' Hcs') as (H1 & H2 & H3 & H4 & H5 & _).
    exact (conj H1 (conj H2 (conj H3 (conj H4 H5)))).
  - refine (conj _ (conj Hpl (conj Hnu Hrk))).
    intros id cs' Hcs'.
    destruct (Hcomp id cs' Hcs') as (H1 & H2 & H3 & H4 & H5 & H6).
    assert (Hk : known_arity_call kind = false)
      by (destruct (known_arity_call kind); [ | reflexivity ];
          discriminate (Hkk eq_refl)).
    exact (conj H1 (conj H2 (conj H3 (conj H4 (conj H5 (H6 Hk)))))).
Qed.

Lemma ind_shape_targets (kind kind' : call_kind) (kps : list pdesc)
    (AA RA : arity) :
  (known_arity_call kind = true -> known_arity_kind kind' = true) ->
  length kps = length (unarize RA) ->
  forallb2 result_ok kps (ind_results kps (unarize RA)) = true ->
  forallb
    (fun '(id, md) =>
       ind_code_ok_o sol cannot_change code_deps unit_code_ids unboxed kind
         kps RA id md)
    codes
  = true ->
  ind_targets_k CSf kind'
    (Code_shape (map (fun _ => Psh_plain) (unarize AA))
       (flat_map pshape_of (ind_results kps (unarize RA)))
       (flat_map kept_kind (ind_results kps (unarize RA))) AA RA
       (pshape1 (Pdesc (ind_exn_decision sol code_deps codes unboxed) true
                   Value))
       true).
Proof.
  intros Hkk Hklen Hres Hall.
  apply (ind_targets_k_gen CSf kind kind' _ Hkk).
  - intros id cs' Hcs'. unfold CSf in Hcs'.
    destruct (lookup id codes) as [md | ] eqn:Hmd; [ | discriminate Hcs' ].
    destruct (code_exn_decision sol code_deps unboxed id) as [d | ] eqn:Hd;
      [ | discriminate Hcs' ].
    assert (Hin : (id, md) ∈ codes)
      by exact (keyed_lookup_in Pos.eqb Pos.eqb_eq id md codes Hmd).
    pose proof (proj1 (forallb_forall _ codes) Hall (id, md) Hin) as Hok.
    cbn beta iota in Hok. unfold ind_code_ok_o in Hok. rewrite Hd in Hok.
    destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed
                id md)
      as [ | pd rd] eqn:Hnc; [ | discriminate Hok ].
    destruct (code_pdescs sol cannot_change code_deps unit_code_ids unboxed
                id md)
      as [[pds rds] | ] eqn:Hpd; [ | discriminate Hok ].
    injection Hcs' as <-.
    assert (Hncc : code_calling_convention_change sol cannot_change
                     code_deps unit_code_ids id md
                   = Not_changing_calling_convention).
    { unfold code_change_u in Hnc. unfold code_calling_convention_change.
      destruct (cannot_change_calling_convention cannot_change unit_code_ids
                  id);
        [ reflexivity | ].
      destruct (lookup id code_deps); [ discriminate Hnc | reflexivity ]. }
    unfold code_pdescs in Hpd. rewrite Hnc, Hncc in Hpd.
    destruct (lookup id code_deps) as [cd | ] eqn:Hcd; [ | discriminate Hpd ].
    destruct (cm_is_tupled md
              || existsb is_unbox
                   (fixed_arity_decisions sol unboxed (cd_return cd)
                      (unarize (cm_result_arity md))))
      eqn:Htup; [ discriminate Hpd | ].
    apply orb_false_iff in Htup as [_ Hnu].
    injection Hpd as Hpds Hrds. subst pds rds.
    pose proof (fixed_rds_kept _ _ Hnu) as Hkept.
    set (rd := fixed_arity_decisions sol unboxed (cd_return cd)
                 (unarize (cm_result_arity md))) in *.
    set (rds := pdescs_of rd (map (fun _ => true) rd)
                  (unarize (cm_result_arity md))) in *.
    apply andb_true_iff in Hok as [Hok Hone].
    apply andb_true_iff in Hok as [Hrlen Hfit].
    apply Nat.eqb_eq in Hrlen.
    split; [ reflexivity | ].
    split; [ cbn [cs_params]; rewrite plain_shapes;
             apply Forall_forall; intros o Ho;
             apply in_map_iff in Ho as (? & <- & _); reflexivity | ].
    split.
    { cbn [cs_result_kinds cs_result_arity].
      rewrite (kept_results_kinds rds Hkept).
      exact (pdescs_of_kinds _ _ _ Hrlen). }
    split.
    { cbn [cs_exn]. destruct (code_exn_decision_cases id d Hd) as [Hdc _].
      unfold ind_exn_decision.
      match goal with
      | |- context [forallb ?g codes] => destruct (forallb g codes) eqn:Hfa
      end;
        destruct Hdc as [-> | ->]; unfold pshape1; cbn [pd_decision];
        try constructor.
      exfalso. pose proof (proj1 (forallb_forall _ codes) Hfa (id, md) Hin)
        as H. cbn beta iota in H. rewrite Hd in H. discriminate H. }
    split.
    { cbn [cs_result_arity cs_results]. intros Hra.
      rewrite Hra, (proj2 (arity_eqb_eq RA RA) eq_refl) in Hfit.
      exact (ind_fits kps rds (unarize RA) Hfit Hres Hkept). }
    { cbn [cs_result_arity cs_results]. intros Hk Hone'.
      rewrite Hk, Hone',
        (proj2 (arity_eqb_eq [Singleton Value] [Singleton Value]) eq_refl)
        in Hone.
      rewrite Hone' in Hrlen.
      change (length (unarize [Singleton Value])) with 1 in Hrlen.
      clearbody rds.
      destruct rds as [ | r [ | r' rest]]; cbn [length] in Hrlen;
        try discriminate Hrlen.
      inversion Hkept as [ | r0 l0 [Hkeep Hdec] _ ]; subst.
      cbn in Hone.
      cbn [flat_map]. unfold pshape_of. rewrite Hkeep.
      destruct Hdec as [Hdec | Hdec]; rewrite Hdec in Hone |- *;
        cbn in Hone; [ reflexivity | discriminate Hone ]. }
  - apply Forall_forall. intros o Ho. apply in_map_iff in Ho as (? & <- & _).
    reflexivity.
  - apply ind_results_no_unboxed.
  - cbn [cs_result_kinds cs_result_arity].
    exact (ind_results_kinds kps (unarize RA) Hklen).
Qed.

(* What [ind_code_ok_o1] gives of a code of the unit, at the shape of a
   call whose code is not known. *)
Lemma ind_code_shape1 (no : bool) (kps : list pdesc) (RA : arity)
    (id : code_id) (cs' : code_shape) :
  forallb2 result_ok kps (ind_results kps (unarize RA)) = true ->
  forallb
    (fun '(id, md) =>
       ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed no
         kps RA id md)
    codes
  = true ->
  CSf id = Some cs' ->
  cs_keeps cs' = true
  ∧ Forall (fun o => o = Psh_plain) (cs_params cs')
  ∧ cs_result_kinds cs' = unarize (cs_result_arity cs')
  ∧ fits (cs_exn cs')
      (pshape1 (Pdesc (ind_exn_decision sol code_deps codes unboxed) true
                  Value))
  ∧ (cs_result_arity cs' = RA ->
     Forall2 fits (cs_results cs')
       (flat_map pshape_of (ind_results kps (unarize RA))))
  ∧ (no = false -> cs_result_arity cs' = [Singleton Value] ->
     cs_results cs' = [Psh_plain])
  ∧ ∃ md : code_metadata,
      lookup id codes = Some md ∧ cs_params_arity cs' = cm_params_arity md.
Proof.
  intros Hres Hall Hcs'. unfold CSf in Hcs'.
  destruct (lookup id codes) as [md | ] eqn:Hmd; [ | discriminate Hcs' ].
  destruct (code_exn_decision sol code_deps unboxed id) as [d | ] eqn:Hd;
    [ | discriminate Hcs' ].
  assert (Hin : (id, md) ∈ codes)
    by exact (keyed_lookup_in Pos.eqb Pos.eqb_eq id md codes Hmd).
  pose proof (proj1 (forallb_forall _ codes) Hall (id, md) Hin) as Hok.
  cbn beta iota in Hok. unfold ind_code_ok_o1 in Hok. rewrite Hd in Hok.
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed
              id md)
    as [ | pd rd] eqn:Hnc; [ | discriminate Hok ].
  destruct (code_pdescs sol cannot_change code_deps unit_code_ids unboxed
              id md)
    as [[pds rds] | ] eqn:Hpd; [ | discriminate Hok ].
  injection Hcs' as <-.
  assert (Hncc : code_calling_convention_change sol cannot_change
                   code_deps unit_code_ids id md
                 = Not_changing_calling_convention).
  { unfold code_change_u in Hnc. unfold code_calling_convention_change.
    destruct (cannot_change_calling_convention cannot_change unit_code_ids
                id);
      [ reflexivity | ].
    destruct (lookup id code_deps); [ discriminate Hnc | reflexivity ]. }
  unfold code_pdescs in Hpd. rewrite Hnc, Hncc in Hpd.
  destruct (lookup id code_deps) as [cd | ] eqn:Hcd; [ | discriminate Hpd ].
  destruct (cm_is_tupled md
            || existsb is_unbox
                 (fixed_arity_decisions sol unboxed (cd_return cd)
                    (unarize (cm_result_arity md))))
    eqn:Htup; [ discriminate Hpd | ].
  apply orb_false_iff in Htup as [_ Hnu].
  injection Hpd as Hpds Hrds. subst pds rds.
  pose proof (fixed_rds_kept _ _ Hnu) as Hkept.
  set (rd := fixed_arity_decisions sol unboxed (cd_return cd)
               (unarize (cm_result_arity md))) in *.
  set (rds := pdescs_of rd (map (fun _ => true) rd)
                (unarize (cm_result_arity md))) in *.
  apply andb_true_iff in Hok as [Hok Hone].
  apply andb_true_iff in Hok as [Hrlen Hfit].
  apply Nat.eqb_eq in Hrlen.
  split; [ reflexivity | ].
  split; [ cbn [cs_params]; rewrite plain_shapes;
           apply Forall_forall; intros o Ho;
           apply in_map_iff in Ho as (? & <- & _); reflexivity | ].
  split.
  { cbn [cs_result_kinds cs_result_arity].
    rewrite (kept_results_kinds rds Hkept).
    exact (pdescs_of_kinds _ _ _ Hrlen). }
  split.
  { cbn [cs_exn]. destruct (code_exn_decision_cases id d Hd) as [Hdc _].
    unfold ind_exn_decision.
    match goal with
    | |- context [forallb ?g codes] => destruct (forallb g codes) eqn:Hfa
    end;
      destruct Hdc as [-> | ->]; unfold pshape1; cbn [pd_decision];
      try constructor.
    exfalso. pose proof (proj1 (forallb_forall _ codes) Hfa (id, md) Hin)
      as H. cbn beta iota in H. rewrite Hd in H. discriminate H. }
  split.
  { cbn [cs_result_arity cs_results]. intros Hra.
    rewrite Hra, (proj2 (arity_eqb_eq RA RA) eq_refl) in Hfit.
    exact (ind_fits kps rds (unarize RA) Hfit Hres Hkept). }
  split.
  { cbn [cs_result_arity cs_results]. intros Hk Hone'.
    rewrite Hk, Hone',
      (proj2 (arity_eqb_eq [Singleton Value] [Singleton Value]) eq_refl)
      in Hone.
    rewrite Hone' in Hrlen.
    change (length (unarize [Singleton Value])) with 1 in Hrlen.
    clearbody rds.
    destruct rds as [ | r [ | r' rest]]; cbn [length] in Hrlen;
      try discriminate Hrlen.
    inversion Hkept as [ | r0 l0 [Hkeep Hdec] _ ]; subst.
    cbn in Hone.
    cbn [flat_map]. unfold pshape_of. rewrite Hkeep.
    destruct Hdec as [Hdec | Hdec]; rewrite Hdec in Hone |- *;
      cbn in Hone; [ reflexivity | discriminate Hone ]. }
  exists md. split; reflexivity.
Qed.

(* [ind_code_shape1], [ind_code_ok_o1] being asked of the code [id]
   alone. *)
Lemma ind_code_shape1_one (no : bool) (kps : list pdesc) (RA : arity)
    (id : code_id) (cs' : code_shape) :
  forallb2 result_ok kps (ind_results kps (unarize RA)) = true ->
  (∀ md : code_metadata,
     lookup id codes = Some md ->
     ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed no kps
       RA id md
     = true) ->
  CSf id = Some cs' ->
  cs_keeps cs' = true
  ∧ Forall (fun o => o = Psh_plain) (cs_params cs')
  ∧ cs_result_kinds cs' = unarize (cs_result_arity cs')
  ∧ fits (cs_exn cs')
      (pshape1 (Pdesc (ind_exn_decision sol code_deps codes unboxed) true
                  Value))
  ∧ (cs_result_arity cs' = RA ->
     Forall2 fits (cs_results cs')
       (flat_map pshape_of (ind_results kps (unarize RA))))
  ∧ (no = false -> cs_result_arity cs' = [Singleton Value] ->
     cs_results cs' = [Psh_plain])
  ∧ ∃ md : code_metadata,
      lookup id codes = Some md ∧ cs_params_arity cs' = cm_params_arity md.
Proof.
  intros Hres Hall Hcs'. unfold CSf in Hcs'.
  destruct (lookup id codes) as [md | ] eqn:Hmd; [ | discriminate Hcs' ].
  destruct (code_exn_decision sol code_deps unboxed id) as [d | ] eqn:Hd;
    [ | discriminate Hcs' ].
  assert (Hin : (id, md) ∈ codes)
    by exact (keyed_lookup_in Pos.eqb Pos.eqb_eq id md codes Hmd).
  pose proof (Hall md eq_refl) as Hok.
  unfold ind_code_ok_o1 in Hok. rewrite Hd in Hok.
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed
              id md)
    as [ | pd rd] eqn:Hnc; [ | discriminate Hok ].
  destruct (code_pdescs sol cannot_change code_deps unit_code_ids unboxed
              id md)
    as [[pds rds] | ] eqn:Hpd; [ | discriminate Hok ].
  injection Hcs' as <-.
  assert (Hncc : code_calling_convention_change sol cannot_change
                   code_deps unit_code_ids id md
                 = Not_changing_calling_convention).
  { unfold code_change_u in Hnc. unfold code_calling_convention_change.
    destruct (cannot_change_calling_convention cannot_change unit_code_ids
                id);
      [ reflexivity | ].
    destruct (lookup id code_deps); [ discriminate Hnc | reflexivity ]. }
  unfold code_pdescs in Hpd. rewrite Hnc, Hncc in Hpd.
  destruct (lookup id code_deps) as [cd | ] eqn:Hcd; [ | discriminate Hpd ].
  destruct (cm_is_tupled md
            || existsb is_unbox
                 (fixed_arity_decisions sol unboxed (cd_return cd)
                    (unarize (cm_result_arity md))))
    eqn:Htup; [ discriminate Hpd | ].
  apply orb_false_iff in Htup as [_ Hnu].
  injection Hpd as Hpds Hrds. subst pds rds.
  pose proof (fixed_rds_kept _ _ Hnu) as Hkept.
  set (rd := fixed_arity_decisions sol unboxed (cd_return cd)
               (unarize (cm_result_arity md))) in *.
  set (rds := pdescs_of rd (map (fun _ => true) rd)
                (unarize (cm_result_arity md))) in *.
  apply andb_true_iff in Hok as [Hok Hone].
  apply andb_true_iff in Hok as [Hrlen Hfit].
  apply Nat.eqb_eq in Hrlen.
  split; [ reflexivity | ].
  split; [ cbn [cs_params]; rewrite plain_shapes;
           apply Forall_forall; intros o Ho;
           apply in_map_iff in Ho as (? & <- & _); reflexivity | ].
  split.
  { cbn [cs_result_kinds cs_result_arity].
    rewrite (kept_results_kinds rds Hkept).
    exact (pdescs_of_kinds _ _ _ Hrlen). }
  split.
  { cbn [cs_exn]. destruct (code_exn_decision_cases id d Hd) as [Hdc _].
    unfold ind_exn_decision.
    match goal with
    | |- context [forallb ?g codes] => destruct (forallb g codes) eqn:Hfa
    end;
      destruct Hdc as [-> | ->]; unfold pshape1; cbn [pd_decision];
      try constructor.
    exfalso. pose proof (proj1 (forallb_forall _ codes) Hfa (id, md) Hin)
      as H. cbn beta iota in H. rewrite Hd in H. discriminate H. }
  split.
  { cbn [cs_result_arity cs_results]. intros Hra.
    rewrite Hra, (proj2 (arity_eqb_eq RA RA) eq_refl) in Hfit.
    exact (ind_fits kps rds (unarize RA) Hfit Hres Hkept). }
  split.
  { cbn [cs_result_arity cs_results]. intros Hk Hone'.
    rewrite Hk, Hone',
      (proj2 (arity_eqb_eq [Singleton Value] [Singleton Value]) eq_refl)
      in Hone.
    rewrite Hone' in Hrlen.
    change (length (unarize [Singleton Value])) with 1 in Hrlen.
    clearbody rds.
    destruct rds as [ | r [ | r' rest]]; cbn [length] in Hrlen;
      try discriminate Hrlen.
    inversion Hkept as [ | r0 l0 [Hkeep Hdec] _ ]; subst.
    cbn in Hone.
    cbn [flat_map]. unfold pshape_of. rewrite Hkeep.
    destruct Hdec as [Hdec | Hdec]; rewrite Hdec in Hone |- *;
      cbn in Hone; [ reflexivity | discriminate Hone ]. }
  exists md. split; reflexivity.
Qed.

(* The shape at which a call whose code is not known is made, on the
   fragment of [calls_expr_o1]: that of [ind_shape_targets], or, for a
   call with one group of arguments in a unit whose code has parameters,
   [ind_targets_1]. *)
Lemma ind_shape_targets1 (ap : apply_expr) (kind' : call_kind)
    (kps : list pdesc) :
  (known_arity_call (ap_call_kind ap) = true ->
   known_arity_kind kind' = true) ->
  length kps = length (unarize (ap_return_arity ap)) ->
  forallb2 result_ok kps (ind_results kps (unarize (ap_return_arity ap)))
  = true ->
  forallb
    (fun '(id, md) =>
       ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
         (never_over codes ap) kps (ap_return_arity ap) id md)
    codes
  = true ->
  let cs :=
    Code_shape (map (fun _ => Psh_plain) (unarize (ap_args_arity ap)))
      (flat_map pshape_of (ind_results kps (unarize (ap_return_arity ap))))
      (flat_map kept_kind (ind_results kps (unarize (ap_return_arity ap))))
      (ap_args_arity ap) (ap_return_arity ap)
      (pshape1 (Pdesc (ind_exn_decision sol code_deps codes unboxed) true
                  Value))
      true
  in
  ind_targets_k CSf kind' cs ∨ ind_targets_1 CSf cs.
Proof.
  intros Hkk Hklen Hres Hall cs.
  assert (Hpl : Forall (fun o => o = Psh_plain) (cs_params cs)).
  { apply Forall_forall. intros o Ho. apply in_map_iff in Ho as (? & <- & _).
    reflexivity. }
  assert (Hnu : no_unboxed (cs_results cs)) by apply ind_results_no_unboxed.
  assert (Hrk : cs_result_kinds cs = unarize (cs_result_arity cs)).
  { cbn [cs cs_result_kinds cs_result_arity].
    exact (ind_results_kinds kps (unarize (ap_return_arity ap)) Hklen). }
  pose proof (fun id cs' =>
                ind_code_shape1 (never_over codes ap) kps (ap_return_arity ap)
                  id cs' Hres Hall)
    as Hcode.
  destruct (never_over codes ap) eqn:Hno.
  - unfold never_over in Hno.
    destruct (known_arity_call (ap_call_kind ap)) eqn:Hk.
    + left.
      apply (ind_targets_k_gen CSf (ap_call_kind ap) kind' cs
               (fun _ => Hkk eq_refl));
        [ | exact Hpl | exact Hnu | exact Hrk ].
      intros id cs' Hcs'.
      destruct (Hcode id cs' Hcs') as (H1 & H2 & H3 & H4 & H5 & _ & _).
      refine (conj H1 (conj H2 (conj H3 (conj H4 (conj H5 _))))).
      intros Hf. rewrite ?Hk in Hf. discriminate Hf.
    + right. cbn [orb] in Hno. apply andb_true_iff in Hno as [Hone Hpar].
      apply Nat.eqb_eq in Hone.
      split; [ exact Hone | ].
      split.
      * refine (conj _ (conj Hpl (conj Hnu Hrk))).
        intros id cs' Hcs'.
        destruct (Hcode id cs' Hcs') as (H1 & H2 & H3 & H4 & H5 & _ & _).
        exact (conj H1 (conj H2 (conj H3 (conj H4 H5)))).
      * intros id cs' Hcs'.
        destruct (Hcode id cs' Hcs') as (_ & _ & _ & _ & _ & _ & md & Hmd
                                         & Hpa).
        rewrite Hpa.
        assert (Hin : (id, md) ∈ codes)
          by exact (keyed_lookup_in Pos.eqb Pos.eqb_eq id md codes Hmd).
        unfold codes_have_params in Hpar.
        pose proof (proj1 (forallb_forall _ codes) Hpar (id, md) Hin) as H.
        cbn beta iota in H. apply negb_true_iff, Nat.eqb_neq in H.
        intros Hnil. rewrite Hnil in H. exact (H eq_refl).
  - left. apply (ind_targets_k_gen CSf (ap_call_kind ap) kind' cs Hkk);
      [ | exact Hpl | exact Hnu | exact Hrk ].
    intros id cs' Hcs'.
    destruct (Hcode id cs' Hcs') as (H1 & H2 & H3 & H4 & H5 & H6 & _).
    exact (conj H1 (conj H2 (conj H3 (conj H4 (conj H5
             (fun _ => H6 eq_refl)))))).
Qed.

(* The shape at which a call whose code is not known is made, on the
   fragment of [calls_expr_f3]: that of [ind_shape_targets1], or, for a
   call of [f] that over-applies nothing, [ind_targets_f], at the code that
   the flow may bring to [f]. *)
Lemma ind_shape_targets_f (ap : apply_expr) (kind' : call_kind)
    (kps : list pdesc) (f : variable) (callee : simple) :
  (known_arity_call (ap_call_kind ap) = true ->
   known_arity_kind kind' = true) ->
  callee = Var f ->
  length kps = length (unarize (ap_return_arity ap)) ->
  forallb2 result_ok kps (ind_results kps (unarize (ap_return_arity ap)))
  = true ->
  forallb
    (fun '(id, md) =>
       never_over codes ap
       && negb (reach (known_arity_call (ap_call_kind ap)) f id)
       || ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
            (never_over codes ap) kps (ap_return_arity ap) id md)
    codes
  = true ->
  let cs :=
    Code_shape (map (fun _ => Psh_plain) (unarize (ap_args_arity ap)))
      (flat_map pshape_of (ind_results kps (unarize (ap_return_arity ap))))
      (flat_map kept_kind (ind_results kps (unarize (ap_return_arity ap))))
      (ap_args_arity ap) (ap_return_arity ap)
      (pshape1 (Pdesc (ind_exn_decision sol code_deps codes unboxed) true
                  Value))
      true
  in
  ind_targets_k CSf kind' cs ∨ ind_targets_1 CSf cs
  ∨ ∃ b : bool, ind_targets_f CSf RCHf b kind' callee cs.
Proof.
  intros Hkk Hcallee Hklen Hres Hall cs.
  destruct (never_over codes ap) eqn:Hno.
  - right. right. exists (known_arity_call (ap_call_kind ap)).
    split.
    { destruct (known_arity_call (ap_call_kind ap)) eqn:Hk;
        [ exact (Hkk eq_refl) | ].
      unfold never_over in Hno. rewrite Hk in Hno. cbn [orb] in Hno.
      apply andb_true_iff in Hno as [Hone _]. apply Nat.eqb_eq in Hone.
      exact Hone. }
    split.
    { exists f. split; [ exact Hcallee | ].
      intros id cs' Hreach Hcs'.
      assert (Hone : ∀ md : code_metadata,
                lookup id codes = Some md ->
                ind_code_ok_o1 sol cannot_change code_deps unit_code_ids
                  unboxed (never_over codes ap) kps (ap_return_arity ap) id md
                = true).
      { intros md Hmd.
        assert (Hin : (id, md) ∈ codes)
          by exact (keyed_lookup_in Pos.eqb Pos.eqb_eq id md codes Hmd).
        pose proof (proj1 (forallb_forall _ codes) Hall (id, md) Hin) as H.
        cbn beta iota in H. unfold RCHf in Hreach.
        rewrite ?Hno, Hreach in H. cbn [andb negb orb] in H.
        rewrite Hno. exact H. }
      destruct (ind_code_shape1_one _ kps (ap_return_arity ap) id cs' Hres
                  Hone Hcs')
        as (H1 & H2 & H3 & H4 & H5 & _ & md & Hmd & Hpa).
      split; [ exact (conj H1 (conj H2 (conj H3 (conj H4 H5)))) | ].
      intros Hk. rewrite Hpa.
      unfold never_over in Hno. rewrite Hk in Hno. cbn [orb] in Hno.
      apply andb_true_iff in Hno as [_ Hpar].
      assert (Hin : (id, md) ∈ codes)
        by exact (keyed_lookup_in Pos.eqb Pos.eqb_eq id md codes Hmd).
      unfold codes_have_params in Hpar.
      pose proof (proj1 (forallb_forall _ codes) Hpar (id, md) Hin) as H.
      cbn beta iota in H. apply negb_true_iff, Nat.eqb_neq in H.
      intros Hnil. rewrite Hnil in H. exact (H eq_refl). }
    split.
    { apply Forall_forall. intros o Ho.
      apply in_map_iff in Ho as (? & <- & _). reflexivity. }
    split; [ apply ind_results_no_unboxed | ].
    cbn [cs cs_result_kinds cs_result_arity].
    exact (ind_results_kinds kps (unarize (ap_return_arity ap)) Hklen).
  - assert (Hall' :
      forallb
        (fun '(id, md) =>
           ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
             (never_over codes ap) kps (ap_return_arity ap) id md)
        codes
      = true).
    { rewrite forallb_forall in Hall |- *. intros [id md] Hin.
      pose proof (Hall _ Hin) as H. cbn beta iota in H |- *.
      rewrite ?Hno in H. cbn [andb orb] in H. rewrite Hno. exact H. }
    destruct (ind_shape_targets1 ap kind' kps Hkk Hklen Hres Hall')
      as [H | H];
      [ left; exact H | right; left; exact H ].
Qed.

(* The shape at which a call whose code is not known is made, on the
   fragment of [calls_expr_f3]: that of [ind_shape_targets_f] for a call
   that over-applies nothing, and [ind_targets_o] otherwise, at the code
   that a call may over-apply. *)
Lemma ind_shape_targets_o2 (ap : apply_expr) (kind' : call_kind)
    (kps : list pdesc) (f : variable) (callee : simple) :
  (known_arity_call (ap_call_kind ap) = true ->
   known_arity_kind kind' = true) ->
  callee = Var f ->
  length kps = length (unarize (ap_return_arity ap)) ->
  forallb2 result_ok kps (ind_results kps (unarize (ap_return_arity ap)))
  = true ->
  forallb
    (fun '(id, md) =>
       never_over codes ap
       && negb (reach (known_arity_call (ap_call_kind ap)) f id)
       || ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
            (never_over codes ap || negb (ovr id)) kps (ap_return_arity ap)
            id md)
    codes
  = true ->
  let cs :=
    Code_shape (map (fun _ => Psh_plain) (unarize (ap_args_arity ap)))
      (flat_map pshape_of (ind_results kps (unarize (ap_return_arity ap))))
      (flat_map kept_kind (ind_results kps (unarize (ap_return_arity ap))))
      (ap_args_arity ap) (ap_return_arity ap)
      (pshape1 (Pdesc (ind_exn_decision sol code_deps codes unboxed) true
                  Value))
      true
  in
  ind_targets_k CSf kind' cs ∨ ind_targets_1 CSf cs
  ∨ (∃ b : bool, ind_targets_f CSf RCHf b kind' callee cs)
  ∨ ind_targets_o CSf OVf cs.
Proof.
  intros Hkk Hcallee Hklen Hres Hall cs.
  destruct (never_over codes ap) eqn:Hno.
  - assert (Hall' :
      forallb
        (fun '(id, md) =>
           never_over codes ap
           && negb (reach (known_arity_call (ap_call_kind ap)) f id)
           || ind_code_ok_o1 sol cannot_change code_deps unit_code_ids
                unboxed (never_over codes ap) kps (ap_return_arity ap) id md)
        codes
      = true).
    { rewrite forallb_forall in Hall |- *. intros [id md] Hin.
      pose proof (Hall _ Hin) as H. cbn beta iota in H |- *.
      rewrite ?Hno in H. rewrite Hno. cbn [andb orb] in H |- *. exact H. }
    destruct (ind_shape_targets_f ap kind' kps f callee Hkk Hcallee Hklen
                Hres Hall')
      as [H | [H | H]];
      [ left; exact H | right; left; exact H | right; right; left; exact H ].
  - right. right. right.
    assert (Hone : ∀ (id : code_id) (md : code_metadata),
              lookup id codes = Some md ->
              ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
                (negb (ovr id)) kps (ap_return_arity ap) id md
              = true).
    { intros id md Hmd.
      assert (Hin : (id, md) ∈ codes)
        by exact (keyed_lookup_in Pos.eqb Pos.eqb_eq id md codes Hmd).
      pose proof (proj1 (forallb_forall _ codes) Hall (id, md) Hin) as H.
      cbn beta iota in H. rewrite ?Hno in H. cbn [andb orb] in H. exact H. }
    split.
    + split.
      * intros id cs' Hcs'.
        destruct (ind_code_shape1_one _ kps (ap_return_arity ap) id cs' Hres
                    (Hone id) Hcs')
          as (H1 & H2 & H3 & H4 & H5 & _).
        exact (conj H1 (conj H2 (conj H3 (conj H4 H5)))).
      * split.
        { apply Forall_forall. intros o Ho.
          apply in_map_iff in Ho as (? & <- & _). reflexivity. }
        split; [ apply ind_results_no_unboxed | ].
        cbn [cs cs_result_kinds cs_result_arity].
        exact (ind_results_kinds kps (unarize (ap_return_arity ap)) Hklen).
    + intros id cs' Hcs' Hov.
      destruct (ind_code_shape1_one _ kps (ap_return_arity ap) id cs' Hres
                  (Hone id) Hcs')
        as (_ & _ & _ & _ & _ & H6 & _).
      apply H6. unfold OVf in Hov. rewrite Hov. reflexivity.
Qed.

(* The shape at which a call whose code is not known is made, on the
   fragment of [calls_expr_f3]: that of [ind_shape_targets_f] for a call
   that over-applies nothing, and [ind_targets_fo] otherwise, at the code
   that the flow may bring to its callee and that what is left of an
   over-application may call. *)
Lemma ind_shape_targets_o3 (ap : apply_expr) (kind' : call_kind)
    (kps : list pdesc) (f : variable) (callee : simple) :
  (known_arity_call (ap_call_kind ap) = true ->
   known_arity_kind kind' = true) ->
  callee = Var f ->
  length kps = length (unarize (ap_return_arity ap)) ->
  forallb2 result_ok kps (ind_results kps (unarize (ap_return_arity ap)))
  = true ->
  forallb
    (fun '(id, md) =>
       never_over codes ap
       && negb (reach (known_arity_call (ap_call_kind ap)) f id)
       || negb (never_over codes ap) && negb (reach false f id || rem id)
       || ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
            (never_over codes ap || negb (ovr id)) kps (ap_return_arity ap)
            id md)
    codes
  = true ->
  let cs :=
    Code_shape (map (fun _ => Psh_plain) (unarize (ap_args_arity ap)))
      (flat_map pshape_of (ind_results kps (unarize (ap_return_arity ap))))
      (flat_map kept_kind (ind_results kps (unarize (ap_return_arity ap))))
      (ap_args_arity ap) (ap_return_arity ap)
      (pshape1 (Pdesc (ind_exn_decision sol code_deps codes unboxed) true
                  Value))
      true
  in
  ind_targets_k CSf kind' cs ∨ ind_targets_1 CSf cs
  ∨ (∃ b : bool, ind_targets_f CSf RCHf b kind' callee cs)
  ∨ ind_targets_o CSf OVf cs ∨ ind_targets_fo CSf RCHf OVf REMf callee cs.
Proof.
  intros Hkk Hcallee Hklen Hres Hall cs.
  destruct (never_over codes ap) eqn:Hno.
  - assert (Hall' :
      forallb
        (fun '(id, md) =>
           never_over codes ap
           && negb (reach (known_arity_call (ap_call_kind ap)) f id)
           || ind_code_ok_o1 sol cannot_change code_deps unit_code_ids
                unboxed (never_over codes ap) kps (ap_return_arity ap) id md)
        codes
      = true).
    { rewrite forallb_forall in Hall |- *. intros [id md] Hin.
      pose proof (Hall _ Hin) as H. cbn beta iota in H |- *.
      rewrite ?Hno in H. rewrite Hno.
      destruct (reach (known_arity_call (ap_call_kind ap)) f id);
        cbn [andb orb negb] in H |- *; exact H. }
    destruct (ind_shape_targets_f ap kind' kps f callee Hkk Hcallee Hklen
                Hres Hall')
      as [H | [H | H]];
      [ left; exact H | right; left; exact H | right; right; left; exact H ].
  - right. right. right. right.
    assert (Hone : ∀ (id : code_id) (md : code_metadata),
              lookup id codes = Some md ->
              reach false f id = true ∨ rem id = true ->
              ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
                (negb (ovr id)) kps (ap_return_arity ap) id md
              = true).
    { intros id md Hmd Hrr.
      assert (Hin : (id, md) ∈ codes)
        by exact (keyed_lookup_in Pos.eqb Pos.eqb_eq id md codes Hmd).
      pose proof (proj1 (forallb_forall _ codes) Hall (id, md) Hin) as H.
      cbn beta iota in H. rewrite ?Hno in H.
      assert (Hr : reach false f id || rem id = true)
        by (destruct Hrr as [-> | ->]; [ reflexivity | apply orb_true_r ]).
      rewrite Hr in H. cbn [andb orb negb] in H. exact H. }
    split.
    + exists f. split; [ exact Hcallee | ].
      intros id cs' Hrr Hcs'.
      destruct (ind_code_shape1_one _ kps (ap_return_arity ap) id cs' Hres
                  (fun md Hmd => Hone id md Hmd Hrr) Hcs')
        as (H1 & H2 & H3 & H4 & H5 & H6 & _).
      split; [ exact (conj H1 (conj H2 (conj H3 (conj H4 H5)))) | ].
      intros Hov. apply H6. unfold OVf in Hov. rewrite Hov. reflexivity.
    + split.
      { apply Forall_forall. intros o Ho.
        apply in_map_iff in Ho as (? & <- & _). reflexivity. }
      split; [ apply ind_results_no_unboxed | ].
      cbn [cs cs_result_kinds cs_result_arity].
      exact (ind_results_kinds kps (unarize (ap_return_arity ap)) Hklen).
Qed.

(* A call whose code is not known: both rebuilds give the arguments at the
   mask of the call, the same, and expect every result, as a call of code
   that keeps its calling convention; the call is related at the shape of
   [ind_shape_targets]. *)
Lemma call_rebuilt_ind (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  cinv C en enu ->
  call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed nr reach
    ovr rem (env_kinds en) C ap
  = true ->
  (∀ z, z ∈ variables_in_apply ap -> fresh z) ->
  ap_exn_continuation ap <> ff -> ap_exn_continuation ap <> ffu ->
  code_id_actually_called F sol ap = None ->
  uc_result (K_of C) (live_conts F sol code_deps codes (Apply ap))
    (cc_rebuild en (Apply ap)) (u_rebuild enu (Apply ap)).
Proof.
  intros Hci Hok Hfresh Hff Hffu Hcalled.
  assert (Hf : ∃ f : variable,
             ap_callee ap = Var f
             ∧ ind_call_ok_f3 F sol cannot_change code_deps unit_code_ids codes
                 unboxed reach ovr rem (env_kinds en) C ap f
               = true).
  { unfold call_ok_f3 in Hok. rewrite Hcalled in Hok. revert Hok.
    destruct (ap_callee ap) as [f | sym | cst].
    - destruct (ap_call_kind ap); intros Hok; exists f;
        (split; [ reflexivity | exact Hok ]).
    - destruct (ap_call_kind ap); intros Hok; discriminate Hok.
    - destruct (ap_call_kind ap); intros Hok; discriminate Hok. }
  destruct Hf as (f & Hcal & Hind).
  unfold ind_call_ok_f3 in Hind.
  apply andb_true_iff in Hind as [Hind Hret].
  apply andb_true_iff in Hind as [Hind Hexn].
  apply andb_true_iff in Hind as [Hind Hmask].
  apply andb_true_iff in Hind as [Hind Halen].
  apply andb_true_iff in Hind as [Hind Hnr].
  apply negb_true_iff in Hnr.
  change (live_conts F sol code_deps codes (Apply ap))
    with (if apply_never_returns F sol code_deps codes ap
          then [ap_exn_continuation ap]
          else free_continuations_of_expr (Apply ap)).
  rewrite Hnr.
  apply andb_true_iff in Hind as [Hfplain Hfused].
  apply Nat.eqb_eq in Halen.
  unfold masked_args_ok in Hmask.
  apply andb_true_iff in Hmask as [Hmeq Hmused].
  apply forallb2_eqb_eq in Hmeq.
  set (dv := ind_exn_decision sol code_deps codes unboxed) in *.
  assert (Hdv : dv = Decided (Keep Value) ∨ dv = Keep_unused Value).
  { unfold dv, ind_exn_decision.
    match goal with
    | |- context [forallb ?g codes] => destruct (forallb g codes)
    end; [ left | right ]; reflexivity. }
  destruct (exn_rebuilt en enu C ap dv Hci Hdv Hexn)
    as (extra_u & Hxu & Hexrel).
  { intros z Hz. apply Hfresh. unfold variables_in_apply.
    apply in_or_app. right. apply in_or_app. right. exact Hz. }
  assert (Hfargs : ∀ z, z ∈ variables_in_simples (ap_args ap) -> fresh z).
  { intros z Hz. apply Hfresh. unfold variables_in_apply.
    apply in_or_app. right. apply in_or_app. left. exact Hz. }
  assert (Hfresh_f : fresh f)
    by (apply Hfresh; unfold variables_in_apply; rewrite Hcal; left;
        reflexivity).
  assert (Hac : apply_change_u F sol cannot_change code_deps unit_code_ids
                  codes unboxed ap
                = Not_changing_u)
    by (unfold apply_change_u; rewrite Hcalled; reflexivity).
  destruct (lookup (ap_continuation ap) C) as [[kps [ | ]] | ] eqn:HCk;
    try discriminate Hret.
  apply andb_true_iff in Hret as [Hret Hcodes_ok].
  apply andb_true_iff in Hret as [Hklen Hres].
  apply Nat.eqb_eq in Hklen.
  assert (Hfx : fields_of unboxed f = None).
  { cbn in Hfplain. apply negb_true_iff in Hfplain.
    unfold Keyed.is_bound in Hfplain. unfold fields_of.
    destruct (Keyed.lookup _ _ _); [ discriminate Hfplain | reflexivity ]. }
  assert (Hdead_eq :
    simple_is_dead sol (ue_env enu) (ap_callee ap)
    || existsb (simple_is_dead sol (ue_env enu)) (ap_args ap)
    = simple_is_dead sol en (ap_callee ap)
      || existsb (simple_is_dead sol en) (ap_args ap)).
  { f_equal; [ apply simple_is_dead_kinds, (ci_kinds _ _ _ Hci) | ].
    clear -Hci. induction (ap_args ap) as [ | a args IH]; [ reflexivity | ].
    simpl. rewrite IH, (simple_is_dead_kinds _ _ a (ci_kinds _ _ _ Hci)).
    reflexivity. }
  unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
  unfold rebuild_apply_cc, apply_calling_convention_change.
  rewrite Hcalled.
  unfold rebuild_apply, rebuild_apply_u. rewrite Hac.
  rewrite Hnr. unfold kept_arguments_u, kept_code_mask. rewrite Hcalled.
  cbn beta iota.
  rewrite (rebuild_arguments_mask sol en ap f Hcal),
    (rebuild_arguments_mask sol_u (ue_env enu) ap f Hcal), <- Hmeq,
    (mask_arguments_kinds (ue_env enu) en _ (ci_kinds _ _ _ Hci)).
  rewrite Hdead_eq.
  destruct (simple_is_dead sol en (ap_callee ap)
            || existsb (simple_is_dead sol en) (ap_args ap))
    eqn:Hdead; [ exact (uc_result_invalid _ _) | ].
  rewrite Hxu. cbn beta iota zeta.
  rewrite Hcal.
  change (unboxed_of unboxed (Var f)) with (fields_of unboxed f).
  rewrite Hfx.
  set (m := call_mask F sol f ap) in *.
  set (kinds := unarize (ap_args_arity ap)) in *.
  set (kinds_r := unarize (ap_return_arity ap)) in *.
  set (rds := ind_results kps kinds_r) in *.
  set (cs := Code_shape (map (fun _ => Psh_plain) kinds)
               (flat_map pshape_of rds) (flat_map kept_kind rds)
               (ap_args_arity ap) (ap_return_arity ap)
               (pshape1 (Pdesc dv true Value)) true).
  match goal with
  | |- uc_result _ _ (make_apply_wrapper _ _ ?X)
         (make_apply_wrapper_u _ _ _ ?Y _ _) =>
    set (ap1 := X); set (ap1u := Y)
  end.
  assert (Hcall : call_rel unboxed dead CSf RCHf OVf REMf ap1 ap1u cs).
  { split.
    { right. split; [ reflexivity | ].
      split; [ refine (ind_shape_targets_o3 ap _ kps f _ _ _ Hklen Hres
                         Hcodes_ok);
               [ intros Hk; unfold ap1; cbn [ap_call_kind];
                 exact (rewrite_kind_known _ _ _ _ ap Hk Hcalled)
               | reflexivity ] | ].
      split; reflexivity. }
    split; [ reflexivity | ].
    split; [ exact (plain2_of f Value Hfx Hfresh_f eq_refl Hfused) | ].
    split; [ exact (args_rel2_mask en (ap_args ap) m kinds Halen Hmused
                      Hfargs) | ].
    split; [ apply plain_shapes_no_dead | ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | ].
    split; [ cbn [cs cs_params cs_params_arity ap1 ap_args_arity];
             rewrite plain_shapes_kinds by reflexivity; reflexivity | ].
    reflexivity. }
  assert (Hexn_ok : exn_call_rel unboxed dead (K_of C) (cs_exn cs) ap1 ap1u)
    by exact (Hexrel ap1 ap1u eq_refl eq_refl eq_refl).
  assert (HKk : lookup (ap_continuation ap) (K_of C)
                = Some (Kshape (flat_map pshape_of kps)
                          (flat_map kept_kind kps) true))
    by (rewrite lookup_K_of, HCk; reflexivity).
  pose proof (ci_ok _ _ _ Hci _ kps true HCk) as Hkps_ok.
  assert (Hlen_kr : length kps = length rds)
    by exact (forallb2_length _ _ _ Hres).
  destruct (apply_decisions_kps C en enu _ kps true Hci HCk)
    as [Happ Hmaskk].
  rewrite Hlen_kr in Happ.
  destruct (ind_results_cc kps kinds_r Hklen) as [Hcomb Hfresh_eq].
  fold rds in Hcomb, Hfresh_eq.
  rewrite (make_apply_wrapper_as_cc en ap1).
  2: { destruct Hmaskk as [Hm | [Hm _]];
       [ left; exists (map pd_keep kps);
         split; [ exact Hm | rewrite length_map; exact Hklen ]
       | right; exact Hm ]. }
  change (unarize (ap_return_arity ap1)) with kinds_r. rewrite <- Hcomb.
  apply (uc_result_results (K_of C) ap ap1 ap1u cs (ap_continuation ap) kps
           rds en enu);
    try reflexivity; try assumption.
  - intros H. exact (Hff (eq_sym H)).
  - intros H. exact (Hffu (eq_sym H)).
  - exact (ind_results_ok kps kinds_r Hkps_ok Hres).
  - rewrite (ci_dec _ _ _ Hci _), HCk. reflexivity.
  - apply Hfresh_eq.
Qed.

Lemma call_rebuilt (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  cinv C en enu ->
  call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed nr reach
    ovr rem (env_kinds en) C ap
  = true ->
  (∀ z, z ∈ variables_in_apply ap -> fresh z) ->
  ap_exn_continuation ap <> ff -> ap_exn_continuation ap <> ffu ->
  uc_result (K_of C) (live_conts F sol code_deps codes (Apply ap))
    (cc_rebuild en (Apply ap)) (u_rebuild enu (Apply ap)).
Proof.
  intros Hci Hok Hfresh Hff Hffu.
  destruct (code_id_actually_called F sol ap) as [idc | ] eqn:Hsome.
  2: exact (call_rebuilt_ind en enu C ap Hci Hok Hfresh Hff Hffu Hsome).
  assert (Hsome' : code_id_actually_called F sol ap <> None)
    by (rewrite Hsome; discriminate).
  clear Hsome. rename Hsome' into Hsome.
  destruct (apply_change_u F sol cannot_change code_deps unit_code_ids codes
              unboxed ap)
    as [ | pd0 rd0] eqn:Hac;
    [ exact (call_rebuilt_nc en enu C ap Hci Hok Hfresh Hff Hffu Hsome Hac)
    | ].
  change (live_conts F sol code_deps codes (Apply ap))
    with (if apply_never_returns F sol code_deps codes ap
          then [ap_exn_continuation ap]
          else free_continuations_of_expr (Apply ap)).
  destruct ap as [callee ckind args k exn extra AA RA].
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity] in *.
  set (ap := {| ap_callee := callee; ap_call_kind := ckind; ap_args := args;
                ap_continuation := k; ap_exn_continuation := exn;
                ap_exn_extra_args := extra; ap_args_arity := AA;
                ap_return_arity := RA |}) in *.
  unfold call_ok_f3 in Hok. cbn [ap_call_kind ap_callee ap ap_args
    ap_continuation ap_exn_continuation ap_exn_extra_args ap_args_arity
    ap_return_arity] in Hok.
  assert (Hck : ckind <> Indirect_unknown_arity)
    by (intros ->; apply Hsome; reflexivity).
  destruct callee as [f | sym | cst]; try discriminate Hok.
  destruct (code_id_actually_called F sol ap) as [id | ] eqn:Hcalled;
    [ | exfalso; exact (Hsome eq_refl) ].
  apply andb_true_iff in Hok as [Hok Hrest].
  apply andb_true_iff in Hok as [Hok Hnr].
  apply andb_true_iff in Hok as [Hok Hfused].
  apply andb_true_iff in Hok as [_ Hfplain].
  destruct (lookup id codes) as [md | ] eqn:Hcodes; [ | discriminate Hrest ].
  unfold apply_change_u in Hac. rewrite Hcalled, Hcodes in Hac.
  destruct (code_pdescs sol cannot_change code_deps unit_code_ids unboxed id
              md)
    as [[pds rds] | ] eqn:Hpd; [ | discriminate Hrest ].
  apply andb_true_iff in Hrest as [Hrest Hret].
  apply andb_true_iff in Hrest as [Hrest _].
  apply andb_true_iff in Hrest as [Hrest Hexn].
  apply andb_true_iff in Hrest as [Hrest Hargs].
  apply andb_true_iff in Hrest as [Hrest Hrok].
  apply andb_true_iff in Hrest as [Hrest Hdok].
  apply andb_true_iff in Hrest as [Hrest Hrlen].
  apply andb_true_iff in Hrest as [Hrest Hdlen].
  apply andb_true_iff in Hrest as [Hrest Halen].
  apply andb_true_iff in Hrest as [Hrest Hmcd].
  apply andb_true_iff in Hrest as [Haa Hra].
  apply arity_eqb_eq in Haa, Hra. subst AA RA.
  apply Nat.eqb_eq in Halen, Hdlen, Hrlen.
  destruct (code_exn_decision sol code_deps unboxed id) as [de | ] eqn:Hde;
    [ | discriminate Hexn ].
  destruct (code_exn_decision_cases id de Hde) as (Hdc & _).
  destruct (exn_rebuilt en enu C ap de Hci Hdc Hexn)
    as (extra_u & Hxu & Hexrel).
  { intros z Hz. apply Hfresh. unfold variables_in_apply.
    apply in_or_app. right. apply in_or_app. right. exact Hz. }
  (* The decisions of the code. *)
  pose proof Hpd as Hpd'.
  unfold code_pdescs in Hpd.
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed id
              md)
    as [ | pd rd] eqn:Hchu; [ congruence | ].
  destruct (code_calling_convention_change sol cannot_change code_deps
              unit_code_ids id md)
    as [ | pd_cc rd_cc] eqn:Hchc; [ discriminate Hpd | ].
  injection Hpd as <- <-.
  pose proof Hchu as Hchu'. pose proof Hchc as Hchc'.
  unfold code_change_u in Hchu. unfold code_calling_convention_change in Hchc.
  destruct (cannot_change_calling_convention cannot_change unit_code_ids id);
    [ discriminate Hchu | ].
  destruct (lookup id code_deps) as [cd | ] eqn:Hcd; [ | discriminate Hchu ].
  injection Hchu as Hpd Hrd. injection Hchc as Hpd_cc Hrd_cc.
  assert (Hlink : Forall2 code_link pd pd_cc)
    by (rewrite <- Hpd, <- Hpd_cc; apply decisions_link).
  set (kinds := unarize (cm_params_arity md)) in *.
  set (kinds_r := unarize (cm_result_arity md)) in *.
  assert (Hlen_pd : length pd = length kinds ∧ length pd_cc = length kinds).
  { rewrite pdescs_of_length, length_map in Hdlen.
    rewrite <- Hpd, <- Hpd_cc, decisions_u_length, decisions_of_length'.
    rewrite <- (Forall2_length Hlink) in Hdlen.
    rewrite <- Hpd, decisions_u_length in Hdlen.
    rewrite length_combine in Hdlen |- *. lia. }
  destruct Hlen_pd as [Hlpd Hlpd_cc].
  (* The results. *)
  subst rd rd_cc.
  destruct (rds_decisions (cd_return cd) kinds_r Hrlen)
    as (Hrcc & Hrkinds & Hrdec).
  set (rds := pdescs_of (decisions_u sol unboxed (cd_return cd) kinds_r)
                (map is_kept (decisions_of sol (cd_return cd) kinds_r))
                kinds_r) in *.
  unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
  assert (Hfx : fields_of unboxed f = None).
  { cbn in Hfplain. apply negb_true_iff in Hfplain.
    unfold Keyed.is_bound in Hfplain. unfold fields_of.
    destruct (Keyed.lookup _ _ _); [ discriminate Hfplain | reflexivity ]. }
  assert (Hdead_eq :
    simple_is_dead sol (ue_env enu) (Var f)
    || existsb (simple_is_dead sol (ue_env enu)) args
    = simple_is_dead sol en (Var f) || existsb (simple_is_dead sol en) args).
  { f_equal; [ apply simple_is_dead_kinds, (ci_kinds _ _ _ Hci) | ].
    clear -Hci. induction args as [ | a args IH]; [ reflexivity | ].
    simpl. rewrite IH, (simple_is_dead_kinds _ _ a (ci_kinds _ _ _ Hci)).
    reflexivity. }
  assert (Hnr_ap : apply_never_returns F sol code_deps codes ap
                   = code_never_returns sol code_deps id)
    by (unfold apply_never_returns; rewrite Hcalled, Hcodes; reflexivity).
  unfold rebuild_apply_cc, apply_calling_convention_change.
  rewrite Hcalled. unfold get_calling_convention_change.
  rewrite Hcodes, Hchc'.
  unfold rebuild_apply_u. unfold ap in *.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  rewrite Hdead_eq.
  destruct (simple_is_dead sol en (Var f) || existsb (simple_is_dead sol en)
              args) eqn:Hdead;
    [ exact (uc_result_invalid _ _) | ].
  unfold apply_change_u, apply_my_closure_u. rewrite Hcalled, Hcodes, Hchu'.
  destruct (my_closure_decision_u cannot_change code_deps unit_code_ids
              unboxed id);
    [ discriminate Hmcd | ].
  rewrite (rewrite_call_kind_direct
             {| ap_callee := Var f; ap_call_kind := ckind;
                ap_args := args; ap_continuation := k;
                ap_exn_continuation := exn; ap_exn_extra_args := extra;
                ap_args_arity := cm_params_arity md;
                ap_return_arity := cm_result_arity md |}
             id Hcalled).
  rewrite Hxu. cbn beta iota zeta.
  assert (Hkok : ∀ x, kind_ok (kind_of_variable en x) = true)
    by exact (ci_kinds_ok _ _ _ Hci).
  assert (Hsum : list_sum (map (@length flambda_kind)
                             (unarize_per_parameter (cm_params_arity md)))
                 = length kinds)
    by apply list_sum_unarize.
  rewrite pdescs_of_length, length_map in Halen.
  destruct (call_args_groups en enu (ci_kinds _ _ _ Hci) Hkok
              (map (@length flambda_kind)
                 (unarize_per_parameter (cm_params_arity md)))
              args pd pd_cc kinds)
    as (groups & Hgu & Hargs_rel & Hk1 & Hk2).
  { intros z Hz. apply Hfresh. unfold variables_in_apply.
    cbn [ap_callee ap_args variables_in_simple app]. right.
    apply in_or_app. left. exact Hz. }
  { exact Hlink. }
  { lia. }
  { exact Hlpd. }
  { exact Hlpd_cc. }
  { exact Hsum. }
  { exact Hdok. }
  { exact Hargs. }
  unfold group_by_parameter. rewrite Hgu. rewrite prepend_nil.
  change (unboxed_of unboxed (Var f)) with (fields_of unboxed f).
  rewrite Hfx.
  set (sizes := map (@length flambda_kind)
                  (unarize_per_parameter (cm_params_arity md))) in *.
  set (groups_cc :=
         map (fun '(args, decisions) =>
                get_args_with_kinds sol en decisions args)
           (combine (group_by_sizes sizes args)
              (group_by_sizes sizes pd_cc))) in *.
  rewrite Hnr_ap.
  match goal with
  | |- uc_result _ _ (make_apply_wrapper_cc _ _ ?A _)
         (make_apply_wrapper_u _ _ _ ?B _ _) =>
    set (ap_cc := A); set (ap_u := B)
  end.
  fold kinds_r. rewrite <- Hrdec, <- Hrcc, <- Hrkinds.
  set (cs := Code_shape (flat_map pshape_of
                           (pdescs_of pd (map is_kept pd_cc) kinds))
               (flat_map pshape_of rds) (flat_map kept_kind rds)
               (arity_of_grouped_decisions_u
                  (prepend_first [Decided Delete]
                     (group_by_parameter (cm_params_arity md) pd)))
               (arity_of_decisions_u
                  (decisions_u sol unboxed (cd_return cd) kinds_r))
               (pshape1 (Pdesc de true Value)) false).
  assert (HCS : CSf id = Some cs).
  { unfold CSf. rewrite Hcodes, Hde, Hchu', Hpd'. reflexivity. }
  assert (Hgpd : length (group_by_sizes sizes pd) = length sizes)
    by apply group_by_sizes_length.
  assert (Hcall : call_rel unboxed dead CSf RCHf OVf REMf ap_cc ap_u cs).
  { split;
      [ left; exists id;
        split; [ reflexivity | split; [ reflexivity | exact HCS ] ] | ].
    split; [ reflexivity | ].
    split; [ apply (plain2_of f Value Hfx); [ | reflexivity | exact Hfused ];
             apply Hfresh; left; reflexivity | ].
    split; [ exact Hargs_rel | ].
    split; [ exact (no_dead_code pd pd_cc _ kinds Hlink) | ].
    split.
    { cbn [ap_args_arity ap_u cs cs_params_arity].
      unfold group_by_parameter. fold sizes.
      exact (arity_groups_eq groups (group_by_sizes sizes pd) Hk1). }
    split; [ reflexivity | ].
    split.
    { cbn [ap_args_arity ap_cc cs cs_params_arity]. unfold arity_of_groups.
      unfold arity_of_grouped_decisions_u.
      rewrite !length_map, prepend_first_length, group_by_parameter_length.
      unfold groups_cc. rewrite length_map, length_combine,
        !group_by_sizes_length.
      unfold sizes, unarize_per_parameter. rewrite !length_map. lia. }
    split.
    { cbn [cs cs_params cs_params_arity ap_args_arity ap_cc].
      rewrite unarize_grouped_u, unarize_arity_of_groups, Hk2,
        flat_map_prepend_delete, flat_map_decisions_concat.
      unfold group_by_parameter. fold sizes.
      rewrite group_by_sizes_concat by (rewrite Hlpd; symmetry; exact Hsum).
      reflexivity. }
    reflexivity. }
  assert (Hexn_ok : exn_call_rel unboxed dead (K_of C) (cs_exn cs) ap_cc ap_u)
    by exact (Hexrel ap_cc ap_u eq_refl eq_refl eq_refl).
  destruct (code_never_returns sol code_deps id) eqn:Hnrc.
  { (* A call of code that never returns. *)
    cbn [negb orb] in Hnr.
    refine (uc_result_never (K_of C) _ ap_cc ap_u cs id _ _ _ Hcall Hexn_ok
              eq_refl HCS Hnr _ _ (make_apply_wrapper_cc_cases en ap_cc _) _).
    - intros H. exact (Hff (eq_sym H)).
    - intros H. exact (Hffu (eq_sym H)).
    - intros x Hx. exact Hx. }
  cbn [orb] in Hret.
  destruct (lookup k C) as [[kps [ | ]] | ] eqn:HCk; try discriminate Hret.
  assert (HKk : lookup k (K_of C)
                = Some (Kshape (flat_map pshape_of kps)
                          (flat_map kept_kind kps) true))
    by (rewrite lookup_K_of, HCk; reflexivity).
  pose proof (ci_ok _ _ _ Hci k kps true HCk) as Hkps_ok.
  assert (Hlen_kr : length kps = length rds)
    by exact (forallb2_length _ _ _ Hret).
  destruct (apply_decisions_kps C en enu k kps true Hci HCk)
    as [Happ _].
  rewrite Hlen_kr in Happ.
  set (ap0 := Apply_expr (Var f) ckind args k exn extra
                (cm_params_arity md) (cm_result_arity md)).
  apply (uc_result_results (K_of C) ap0 ap_cc ap_u cs k kps rds en enu);
    try reflexivity; try assumption.
  - intros H. exact (Hff (eq_sym H)).
  - intros H. exact (Hffu (eq_sym H)).
  - rewrite (ci_dec _ _ _ Hci k), HCk. reflexivity.
Qed.

Lemma names_let (x : variable) (kind : flambda_kind) (d : named) (body : expr) :
  names_ok (Let x kind d body) ->
  names_ok body ∧ var_ok x
  ∧ ∀ z, z ∈ variables_in_named d -> var_ok z.
Proof.
  intros [Hv Hk]. split; [ split | split ].
  - intros z Hz. apply Hv. right. apply in_or_app. right. exact Hz.
  - exact Hk.
  - apply Hv. left. reflexivity.
  - intros z Hz. apply Hv. right. apply in_or_app. left. exact Hz.
Qed.

Lemma uc_result_let (K : finite_map kshape) (x : variable)
    (kind : flambda_kind) (d d' : named) (fc0 : list continuation)
    (body body' : expr) :
  plain2 unboxed dead x = true -> named_plain2 unboxed dead d = true ->
  (∀ q args, d <> Prim (Heap q) args) ->
  uc_result K fc0 body body' ->
  d' = d ->
  uc_result K fc0 (Let x kind d body) (Let x kind d' body').
Proof.
  intros Hx Hd Hnh (Huc & Hfc & Hbs & Hincl) ->.
  split; [ apply uc_let; assumption | ].
  cbn [free_continuations_of_expr binds_statics].
  split; [ exact Hfc | split; [ exact Hbs | exact Hincl ] ].
Qed.

Lemma ac_rel2_conts (K : finite_map kshape) (ac ac' : apply_cont) :
  ac_rel2 unboxed dead K ac ac' ->
  apply_cont_continuations ac' = apply_cont_continuations ac.
Proof.
  intros (Hk & Htrap & _). unfold apply_cont_continuations.
  rewrite Hk, Htrap. reflexivity.
Qed.

Lemma map_pdesc_combine (ds : list decision) (keeps : list bool)
    (kinds : list flambda_kind) :
  length ds = length keeps -> length keeps = length kinds ->
  map pd_decision
    (map (fun '(d, keep, kind) => Pdesc d keep kind)
       (combine (combine ds keeps) kinds))
  = ds
  ∧ map pd_keep
      (map (fun '(d, keep, kind) => Pdesc d keep kind)
         (combine (combine ds keeps) kinds))
    = keeps.
Proof.
  revert keeps kinds.
  induction ds as [ | d ds IH]; intros [ | b keeps] [ | kd kinds] H1 H2;
    cbn in H1, H2 |- *; try discriminate H1; try discriminate H2;
    try (split; reflexivity).
  injection H1 as H1. injection H2 as H2.
  destruct (IH keeps kinds H1 H2) as [IH1 IH2].
  rewrite IH1, IH2. split; reflexivity.
Qed.

Lemma handler_pdescs_maps (is_exn : bool) (params : list variable)
    (md : handler_metadata) (body : expr) :
  length params = length (hm_params_kinds md) ->
  map pd_decision (handler_pdescs sol unboxed is_exn (Handler params md body))
  = handler_decisions true sol unboxed is_exn (Handler params md body)
  ∧ map pd_keep (handler_pdescs sol unboxed is_exn (Handler params md body))
    = handler_mask sol is_exn (Handler params md body).
Proof.
  intros Hlen.
  pose proof (with_kinds_combine params (hm_params_kinds md) Hlen)
    as Hbound.
  set (bound := with_kinds params (hm_params_kinds md)) in *.
  set (mask := handler_mask sol is_exn (Handler params md body)).
  assert (Hmlen : length mask = length bound)
    by (unfold mask, handler_mask; apply params_to_keep_length).
  assert (Hblen : length bound = length (hm_params_kinds md))
    by (rewrite Hbound, length_combine, Hlen; apply Nat.min_id).
  assert (Hds : handler_decisions true sol unboxed is_exn
                  (Handler params md body)
                = map decide_param (combine bound mask))
    by reflexivity.
  unfold handler_pdescs. fold mask. rewrite Hds.
  apply map_pdesc_combine.
  - rewrite length_map, length_combine, Hmlen. apply Nat.min_id.
  - rewrite Hmlen. exact Hblen.
Qed.

Lemma rebuild_uc :
  (∀ e : expr, rebuild_ok_expr e)
  ∧ (∀ h : continuation_handler,
       ∀ (en : rebuild_env) (enu : unbox_env)
         (C : finite_map (list pdesc * bool)) (ctx : option code_id)
         (ce : finite_map (list node)) (is_exn : bool),
         cinv C en enu ->
         calls_handler_f3 F sol cannot_change code_deps unit_code_ids codes
           unboxed nr reach ovr rem (env_kinds en) C (handler_pdescs sol unboxed
               is_exn
             h)
           h
         = true ->
         names_ok_h h ->
         FlambdaReaperFlowProofs.handler_edges (graph_of_facts F) noted deps
           cds kept_slots codes0 symbols True False ctx ce h ->
         ∀ dead_check : bool,
           ∃ (ps ps' : list (variable * flambda_kind))
             (md md' : handler_metadata) (hp hu : expr),
             rebuild_handler_cc F sol cannot_change code_deps pdc
               unit_code_ids codes ff en dead_check (handler_mask sol is_exn h)
               h
             = Handler (map fst ps) md hp
             ∧ rebuild_handler_u true F sol cannot_change code_deps pdc
                 unit_code_ids codes unboxed ffu [] enu dead_check
                 (handler_decisions true sol unboxed is_exn h) h
               = Handler (map fst ps') md' hu
             ∧ params_rel2 unboxed dead
                 (flat_map pshape_of (handler_pdescs sol unboxed is_exn h))
                 ps ps'
             ∧ hm_params_kinds md = map snd ps
             ∧ hm_params_kinds md' = map snd ps'
             ∧ hm_params_subkinds md' = []
             ∧ map snd ps
               = flat_map kept_kind (handler_pdescs sol unboxed is_exn h)
             ∧ map pd_decision (handler_pdescs sol unboxed is_exn h)
               = handler_decisions true sol unboxed is_exn h
             ∧ map pd_keep (handler_pdescs sol unboxed is_exn h)
               = handler_mask sol is_exn h
             ∧ uc_result (K_of C) (live_conts_handler F sol code_deps codes h)
                 hp hu).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en enu C ctx ce Hci Hloc Hnames Hedges.
    cbn [calls_expr_f3] in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hkind Hbind].
    destruct (names_let x kind d body Hnames) as (Hnb & [Hxf Hxd] & Hdn).
    destruct Hedges as [Hnamed_edges Hbody_edges].
    assert (Hci' : cinv C (bind_variables en [(x, kind)])
                     (ubind_variables enu [(x, kind)]))
      by (apply cinv_bind; [ exact Hci | intros y k [Heq | []];
                             injection Heq as <- <-; exact Hkind ]).
    specialize (IH (bind_variables en [(x, kind)])
                  (ubind_variables enu [(x, kind)]) C ctx ce Hci'
                  Hbody Hnb Hbody_edges).
    cbn [live_conts]. unfold cc_rebuild, u_rebuild in *.
    cbn [rebuild_expr_cc rebuild_expr_u].
    unfold is_pending_projection.
    rewrite (ci_proj _ _ _ Hci), (ci_proj_u _ _ _ Hci).
    rewrite (rebuild_named_kinds sol en (ue_env enu) x d
               (eq_sym (ci_kinds _ _ _ Hci))).
    assert (Hkok : ∀ y, kind_ok (kind_of_variable (ue_env enu) y) = true).
    { intros y. unfold kind_of_variable. rewrite (ci_kinds _ _ _ Hci).
      exact (ci_kinds_ok _ _ _ Hci y). }
    assert (Hfresh_d : ∀ z, z ∈ variables_in_named d -> fresh z)
      by (intros z Hz; exact (proj1 (Hdn z Hz))).
    destruct (fields_of unboxed x) as [fields | ] eqn:Hx.
    + (* An unboxed variable. *)
      apply (strong_unboxed_binding unboxed sol _ x kind d fields Hbind)
        in Hfresh_d.
      rename Hfresh_d into Hbind'.
      destruct (unboxed_binding_cases sol unboxed _ x kind d fields Hbind')
        as (-> & [(tag & args & -> & Hfields)
                 | [(bn & c & -> & Hc & Hfields)
                   | [(y & fields_y & -> & Hne & Hy & Hflat & Hyused)
                     | (bk & mut & i & y & fy & nested & -> & Hne & Hy & Hny
                        & Halike)]]]);
        cbn beta iota;
        (destruct (is_dead_var sol x Value) eqn:Hdead;
         [ exact (uc_result_invalid _ _) | ]);
        (destruct (is_var_used sol x Value) eqn:Hused;
         [ | exact IH ]);
        change (unboxed_of_node unboxed (Node_variable x))
          with (fields_of unboxed x);
        rewrite Hx.
      * (* A block. *)
        cbn [rebuild_named]. simpl in Hnamed_edges.
        destruct Hnamed_edges as [_ Hnamed_edges].
        destruct (bind_block_fields_lets2 sol unboxed (ue_env enu) x tag args
                    fields)
          with (body := rebuild_expr_u true F sol cannot_change code_deps
                          pdc unit_code_ids codes unboxed ffu []
                          (ubind_variables enu [(x, Value)]) body)
          as (ls & Hl & Hentries).
        { intros f u Hin.
          destruct (Hfields f u Hin) as (i & a & -> & Hnth & Hkind' & Hcase).
          rewrite kinds_env_kind in Hkind'.
          destruct (field_kept F sol unboxed is_local sol_closed local_slots
                      fields_read x fields _ u a (ue_env enu) Hx Hin
                      (proj1 Hnamed_edges i a Hnth) eq_refl)
            as [Hused' Hkept].
          exists i, a. repeat split; try assumption.
          unfold kinds_env in Hkind'. unfold simple_kind in Hkind' |- *.
          destruct a; [ | reflexivity | exact Hkind' ].
          unfold kind_of_variable in Hkind' |- *.
          rewrite (ci_kinds _ _ _ Hci). exact Hkind'. }
        unfold rebuild_unboxed_binding. rewrite Hl.
        destruct IH as (IHuc & IHfc & IHbs & IHincl).
        destruct (lets_free_continuations (concat ls)
                    (rebuild_expr_u true F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body))
          as [Hfc Hbs].
        split.
        { refine (uc_alloc unboxed dead CSf NRf RCHf OVf REMf _ x _ fields
                    ls _ _ Hx Hxf _ _ IHuc).
          - apply (alloc_entries_to2 (ue_env enu)); [ exact Hkok
                                                     | | exact Hentries ].
            intros f kind' s0 z Hf Hs0. subst s0.
            destruct Hf as (j & _ & _ & Hn).
            exact (rebuild_fields_used (ue_env enu) _ args 0 j (Var z) z Hn
                     eq_refl).
          - intros q0 args0 Hq0; discriminate Hq0. }
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; [ exact IHfc | split; [ exact IHbs | exact IHincl ] ].
      * (* A boxed number. *)
        simpl in Hnamed_edges. destruct Hnamed_edges as [_ Hnamed_edges].
        destruct IH as (IHuc & IHfc & IHbs & IHincl).
        destruct (bind_boxed_number_lets unboxed x bn c fields
                    (rebuild_named sol (ue_env enu) x
                       (Prim (Unary (Box_number bn)) [c])) Hc)
          with (body := rebuild_expr_u true F sol cannot_change code_deps
                          pdc unit_code_ids codes unboxed ffu []
                          (ubind_variables enu [(x, Value)]) body)
          as (ls & Hl & Hentries).
        { intros Hne. destruct fields as [ | [f u] fields]; [ easy | ].
          destruct (Hfields f u (or_introl eq_refl)) as (v & -> & ->).
          destruct (field_kept F sol unboxed is_local sol_closed local_slots
                      fields_read x _ _ (Not_unboxed v) c (ue_env enu) Hx
                      (or_introl eq_refl) (proj1 Hnamed_edges) eq_refl)
            as [Hused' Hkept].
          cbn [rebuild_named]. unfold field_used. rewrite Hused'. simpl.
          rewrite (kept_rewrite sol _ c Hkept). reflexivity. }
        { intros f u Hin. exact (Hfields f u Hin). }
        unfold rebuild_unboxed_binding. rewrite Hl.
        destruct (lets_free_continuations (concat ls)
                    (rebuild_expr_u true F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body))
          as [Hfc Hbs].
        split.
        { refine (uc_alloc unboxed dead CSf NRf RCHf OVf REMf _ x _ fields
                    _ _ _ Hx Hxf _ _ IHuc).
          - apply (alloc_entries_to2 (ue_env enu)); [ exact Hkok
                                                     | | exact Hentries ].
            intros f kind' s0 z Hf Hs0. subst s0.
            cbn [rebuild_named] in Hf.
            destruct (field_used sol _ _) in Hf; [ | contradiction ].
            destruct Hf as (_ & _ & Hz).
            exact (rewrite_var_used (ue_env enu) c z (eq_sym Hz)).
          - intros q0 args0 Hq0; cbn [rebuild_named] in Hq0;
              destruct (field_used _ _ _); discriminate Hq0. }
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; [ exact IHfc | split; [ exact IHbs | exact IHincl ] ].
      * (* An alias of an unboxed variable. *)
        destruct IH as (IHuc & IHfc & IHbs & IHincl).
        cbn [rebuild_named rewrite_simple].
        unfold kind_of_variable in Hyused |- *.
        cbn [env_kinds kinds_env] in Hyused.
        rewrite (ci_kinds _ _ _ Hci), Hyused.
        unfold rebuild_unboxed_binding.
        destruct (bind_fields_pairs fields fields_y
                    (rebuild_expr_u true F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body)
                    Hflat)
          as (pairs & Hentries & Hl).
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hl.
        destruct (lets_free_continuations (pairs_lets pairs)
                    (rebuild_expr_u true F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body))
          as [Hfc Hbs].
        split; [ exact (uc_alias unboxed dead CSf NRf RCHf OVf REMf _ x y fields
            fields_y
                          pairs _ _ Hx Hxf Hy Hne Hentries IHuc) | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; [ exact IHfc | split; [ exact IHbs | exact IHincl ] ].
      * (* The load of a field unboxed in turn. *)
        destruct IH as (IHuc & IHfc & IHbs & IHincl).
        cbn [rebuild_named]. unfold rebuild_unboxed_binding, load_unboxed_field.
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hny.
        destruct (bind_fields_pairs fields nested
                    (rebuild_expr_u true F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body)
                    Halike)
          as (pairs & Hentries & Hl).
        rewrite Hl.
        destruct (lets_free_continuations (pairs_lets pairs)
                    (rebuild_expr_u true F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body))
          as [Hfc Hbs].
        split.
        -- apply (uc_load unboxed dead CSf NRf RCHf OVf REMf _ x Value _
                    (Field_block (Z.to_nat i) Value)
                    (rewrite_simple sol (ue_env enu) (Var y)) y fields fy
                    nested pairs); try assumption.
           ++ reflexivity.
           ++ simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
              right. eexists. reflexivity.
           ++ exact (keyed_lookup_in field_eqb field_eqb_decides _ _ fy Hny).
        -- cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
           split; [ exact IHfc | split; [ exact IHbs | exact IHincl ] ].
    + (* A variable that is not unboxed. *)
      assert (Hd : (plain_binding_ok unboxed x kind d = true)
                   ∨ (closure_binding_ok x d body = true)).
      { apply orb_true_iff in Hbind as [Hb | Hb]; [ left | right; exact Hb ].
        exact (strong_plain_binding unboxed x kind d Hb Hfresh_d). }
      destruct Hd as [Hbind' | Hclo].
      * assert (Hd : (∃ s0, d = Simple s0) ∨ (∃ p args, d = Prim p args)).
        { destruct (plain_binding_cases sol unboxed x kind d Hbind')
            as [Hp | (p & y & f & fs & -> & _)]; [ | right; eauto ].
          destruct d; try discriminate Hp; eauto. }
        destruct Hd as [(s0 & ->) | (p & args & ->)]; cbn beta iota;
          (destruct (is_dead_var sol x kind) eqn:Hdead;
           [ exact (uc_result_invalid _ _) | ]);
          (destruct (is_var_used sol x kind) eqn:Hused; [ | exact IH ]);
          change (unboxed_of_node unboxed (Node_variable x))
            with (fields_of unboxed x);
          change (changed_of_node [] (Node_variable x))
            with (@None closure_repr);
          rewrite Hx; cbn iota.
        -- destruct (plain_binding_cases sol unboxed x kind _ Hbind')
             as [Hp | (p & y & f & fs & Heq & _)]; [ | discriminate Heq ].
           rewrite (rebuild_let_u_plain sol unboxed _ x kind _ _ Hp).
           apply uc_result_let; [ | | | exact IH | reflexivity ].
           ++ exact (plain2_of x kind Hx Hxf Hkind Hused).
           ++ apply named_plain2_rebuilt; [ exact Hkok | exact Hp ].
           ++ apply rebuild_named_not_heap.
              eapply FlambdaReaperFlowProofs.named_edges_not_heap.
              exact Hnamed_edges.
        -- destruct (plain_binding_cases sol unboxed x kind _ Hbind')
             as [Hp | (p' & y & f & fs & Heq & Hread & Hreads & Hnamed & Hy
                       & Hfield)].
           ++ rewrite (rebuild_let_u_plain sol unboxed _ x kind _ _ Hp).
              apply uc_result_let; [ | | | exact IH | reflexivity ].
              ** exact (plain2_of x kind Hx Hxf Hkind Hused).
              ** apply named_plain2_rebuilt; [ exact Hkok | exact Hp ].
              ** apply rebuild_named_not_heap.
                 eapply FlambdaReaperFlowProofs.named_edges_not_heap.
                 exact Hnamed_edges.
           ++ rewrite Heq in Hnamed, Hnamed_edges |- *.
              rewrite (Hnamed (ue_env enu)).
              assert (Hacc : g_accessor (graph_of_facts F) (Node_variable x) f
                               (Node_variable y)).
              { clear -Hread Hnamed_edges.
                destruct p'; try discriminate Hread; simpl in Hread;
                  injection Hread as <-; exact (proj1 (proj2 Hnamed_edges)). }
              destruct (reads_found x y kind f fs Hx Hy Hacc
                          (read_of_block_like p' y f Hread) Hdead Hused Hfield)
                as (v & Hv).
              unfold rebuild_let_u.
              assert (Hfr : field_read (Prim (Unary p') [Var y])
                            = Some (f, Var y)).
              { clear -Hread. destruct p'; try discriminate Hread;
                  exact Hread. }
              rewrite Hfr. cbn [unboxed_of].
              change (unboxed_of_node unboxed (Node_variable y))
                with (fields_of unboxed y).
              rewrite Hy, Hv.
              destruct IH as (IHuc & IHfc & IHbs & IHincl).
              split; [ | split; [ exact IHfc | split; [ exact IHbs
                                                      | exact IHincl ] ] ].
              apply (uc_read unboxed dead CSf NRf RCHf OVf REMf _ x kind
                       (Prim (Unary p')
                          [rewrite_simple sol (ue_env enu) (Var y)]) f
                       (rewrite_simple sol (ue_env enu) (Var y)) y v fs).
              ** exact (plain2_of x kind Hx Hxf Hkind Hused).
              ** apply Hreads.
              ** simpl. destruct (is_var_used sol y _);
                   [ left; reflexivity | ].
                 right. eexists. reflexivity.
              ** exact Hy.
              ** exact (keyed_lookup_in field_eqb field_eqb_decides f _ fs Hv).
              ** exact IHuc.
      * (* A closure. *)
        unfold closure_binding_ok in Hclo.
        destruct d as [ | | | slot decls value_slots | | | ];
          try discriminate Hclo.
        destruct value_slots as [ | ? ?]; [ | discriminate Hclo ].
        apply andb_true_iff in Hclo as [_ Hproj].
        destruct (projections_of_set x (Closure_of_set slot decls []) body)
          as [ | ? ?] eqn:Hp; [ | discriminate Hproj ].
        cbn beta iota. cbn [existsb orb].
        destruct (is_dead_var sol x kind) eqn:Hdead;
          [ exact (uc_result_invalid _ _) | ].
        cbn [orb].
        assert (Hun : is_unboxed_name unboxed (Node_variable x) = false).
        { unfold is_unboxed_name, Keyed.is_bound. unfold fields_of in Hx.
          rewrite Hx. reflexivity. }
        rewrite Hun. cbn [orb].
        destruct (is_var_used sol x kind) eqn:Hused; cbn [orb];
          [ | exact IH ].
        unfold rebuild_set_u. change (changed_of_node [] (Node_variable x))
          with (@None closure_repr).
        cbn [rebuild_set rebuild_value_slots flat_map].
        apply uc_result_let; [ | reflexivity | | exact IH | reflexivity ].
        -- exact (plain2_of x kind Hx Hxf Hkind Hused).
        -- intros q0 args0 Hq0. discriminate Hq0.
  - (* Let_static *)
    intros bindings body IHcode IH en enu C ctx ce Hci Hloc
      [Hvars Hconts] Hedges.
    cbn [calls_expr_f3] in Hloc. apply andb_true_iff in Hloc as [Hbs Hbody].
    destruct Hedges as [Hb_edges Hbody_edges].
    cbn [variables_of continuations_of] in Hvars, Hconts.
    destruct (IH (bind_variables en []) (ubind_variables enu []) C
                ctx ce (cinv_bind C en enu [] Hci
                          (fun y k (H : (y, k) ∈ []) => match H with end)))
      as (IHuc & IHfc & IHbs & IHincl).
    { exact Hbody. }
    { split; intros; [ apply Hvars | apply Hconts ]; apply in_or_app; right;
        assumption. }
    { exact Hbody_edges. }
    destruct (bindings_rebuilt en enu C bindings IHcode Hci Hbs)
      as (Hub & Hbfc & Hbbs & Hbincl).
    { intros z Hz. apply Hvars, in_or_app. left. exact Hz. }
    { intros k Hk. apply Hconts, in_or_app. left. exact Hk. }
    { exact Hb_edges. }
    rewrite cc_rebuild_let_static, u_rebuild_let_static.
    split; [ apply uc_static; [ exact Hub | exact IHuc ] | ].
    cbn [free_continuations_of_expr binds_statics live_conts].
    split; [ apply incl_app_app; [ exact Hbfc | exact IHfc ] | ].
    split; [ apply orb_impl2; [ intros H; rewrite Hbbs; exact H
                              | exact IHbs ] | ].
    intros k Hk.
    apply in_app_or in Hk as [Hk | Hk]; apply in_or_app;
      [ left; exact (Hbincl k Hk) | right; exact (IHincl k Hk) ].
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en enu C ctx ce Hci Hloc
      [Hvars Hconts] Hedges.
    cbn [calls_expr_f3] in Hloc. cbv zeta in Hloc.
    apply andb_true_iff in Hloc as [Hh Hbody].
    destruct Hedges as [Hh_edges Hbody_edges].
    cbn [variables_of continuations_of] in Hvars, Hconts.
    set (is_exn := handler_is_exn_handler h) in *.
    set (ps := handler_pdescs sol unboxed is_exn h) in *.
    destruct (IHh en enu C ctx ce is_exn Hci) with (dead_check := true)
      as (ps_c & ps_u & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd'
          & Hsk' & Hkinds & Hdecs & Hmask & Hhres).
    { exact Hh. }
    { split; intros; [ apply Hvars | apply Hconts ];
        [ apply in_or_app; left | right; apply in_or_app; left ];
        assumption. }
    { exact Hh_edges. }
    fold ps in Hdecs, Hmask, Hkinds, Hrel.
    assert (Hps_ok : forallb pdesc_ok ps = true).
    { destruct h as [params md0 handler]. cbn [calls_handler_f3] in Hh.
      apply andb_true_iff in Hh as [Hh _].
      apply andb_true_iff in Hh as [_ Hh]. exact Hh. }
    pose proof (cinv_conts C en enu [(k, (ps, true))] Hci
                  (fun k0 ps0 h0 (H : (k0, (ps0, h0)) ∈ [(k, (ps, true))]) =>
                     match H with
                     | or_introl Heq =>
                       eq_ind_r (fun p => forallb pdesc_ok p = true) Hps_ok
                         (f_equal (fun t => fst (snd t)) (eq_sym Heq))
                     | or_intror H' => match H' with end
                     end))
      as Hci'.
    cbn [map app] in Hci'. rewrite Hdecs, Hmask in Hci'.
    destruct (IH _ _ ((k, (ps, true)) :: C) ctx
                ((k, FlambdaReaperTraversal.handler_params h) :: ce) Hci')
      as (IHuc & IHfc & IHbs & IHincl).
    { exact Hbody. }
    { split; intros; [ apply Hvars | apply Hconts ];
        [ apply in_or_app; right | right; apply in_or_app; right ];
        assumption. }
    { exact Hbody_edges. }
    rewrite cc_rebuild_let_cont, u_rebuild_let_cont. fold is_exn. cbv zeta.
    cbn [live_conts].
    set (pb := cc_rebuild (bind_continuations en
                             [(k, handler_mask sol is_exn h)]) body) in *.
    set (qb := u_rebuild (ubind_continuations enu
                            [(k, handler_decisions true sol unboxed is_exn h)])
                 body) in *.
    destruct (among k (free_continuations_of_expr qb)) eqn:Hkq.
    + (* Both keep the handler. *)
      assert (Hkp : among k (free_continuations_of_expr pb) = true)
        by (apply among_In, IHfc, among_In; exact Hkq).
      rewrite Hkp, Hp, Hu. destruct Hhres as (Hhuc & Hhfc & Hhbs & Hhincl).
      split.
      { apply (uc_let_cont unboxed dead CSf NRf RCHf OVf REMf (K_of C) k
                 (flat_map pshape_of ps) ps_c ps_u md md' hp hu);
          try assumption.
        cbn [K_of map] in IHuc. rewrite Hkinds. exact IHuc. }
      cbn [free_continuations_of_expr free_continuations_of_handler
           binds_statics handler_binds_statics].
      split; [ apply incl_app_app;
               [ exact Hhfc | apply incl_without_mono; exact IHfc ] | ].
      split; [ exact (orb_impl2 _ _ _ _ Hhbs IHbs) | ].
      intros k' Hk'.
      apply in_app_or in Hk' as [Hk' | Hk']; apply in_or_app;
        [ left; exact (Hhincl k' Hk') | right ].
      apply in_without_iff in Hk' as [Hk' Hne].
      apply in_without_iff. split; [ exact (IHincl k' Hk') | exact Hne ].
    + assert (Hknq : k ∉ free_continuations_of_expr qb)
        by (intros Hin; apply among_In in Hin; rewrite Hin in Hkq;
            discriminate Hkq).
      assert (Hincl' : incl (free_continuations_of_expr qb)
                         (without [k] (live_conts F sol code_deps codes body))).
      { intros k' Hk'. apply in_without_iff.
        split; [ exact (IHincl k' Hk') | ].
        intros [Heq | []]. subst k'. exact (Hknq Hk'). }
      destruct (among k (free_continuations_of_expr pb)) eqn:Hkp.
      * (* Only the first keeps the handler: the second dropped it, the call
           that returns to it never returning. *)
        rewrite Hp. destruct Hhres as (_ & _ & Hhbs & _).
        split.
        { apply uc_let_cont_never. cbn [K_of map] in IHuc.
          exact (uc_never_shadow unboxed dead CSf NRf RCHf OVf REMf _ _ _ _
                   _ IHuc Hknq). }
        cbn [free_continuations_of_expr free_continuations_of_handler
             binds_statics handler_binds_statics].
        split; [ | split ].
        -- intros k' Hk'. apply in_or_app. right. apply in_without_iff.
           split; [ exact (IHfc k' Hk') | ].
           intros [Heq | []]. subst k'. exact (Hknq Hk').
        -- intros H. apply orb_true_iff. right. exact (IHbs H).
        -- intros k' Hk'. apply in_or_app. right. exact (Hincl' k' Hk').
      * (* Neither keeps it. *)
        split; [ | split; [ exact IHfc | split; [ exact IHbs | ] ] ].
        -- apply (uc_agree unboxed dead CSf NRf RCHf OVf REMf _ _ _ IHuc).
           intros k' Hk'.
           cbn [K_of map]. rewrite lookup_cons.
           destruct (Pos.eqb k' k) eqn:E; [ | reflexivity ].
           apply Pos.eqb_eq in E. subst k'. exfalso.
           assert (Ht : among k (free_continuations_of_expr pb) = true)
             by (apply among_In; exact Hk').
           rewrite Ht in Hkp. discriminate Hkp.
        -- intros k' Hk'. apply in_or_app. right. exact (Hincl' k' Hk').
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en enu C ctx ce Hci Hloc
      [Hvars Hconts] Hedges.
    cbn [calls_expr_f3] in Hloc. cbv zeta in Hloc.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite forallb_forall in Hhs.
    destruct Hedges as [Hhs_edges Hbody_edges].
    cbn [variables_of continuations_of] in Hvars, Hconts.
    set (D := map (fun '(k, h) =>
                     (k, (handler_pdescs sol unboxed false h, true)))
                handlers) in *.
    assert (Hhs' : ∀ k h, (k, h) ∈ handlers ->
              calls_handler_f3 F sol cannot_change code_deps unit_code_ids codes
                unboxed nr reach ovr rem (env_kinds en) (D ++ C)
                (handler_pdescs sol unboxed false h) h
              = true)
      by (intros k h Hin; exact (Hhs _ Hin)).
    clear Hhs.
    assert (Hmaps : ∀ k h, (k, h) ∈ handlers ->
              map pd_decision (handler_pdescs sol unboxed false h)
              = handler_decisions true sol unboxed false h
              ∧ map pd_keep (handler_pdescs sol unboxed false h)
                = handler_mask sol false h).
    { intros k [params md hbody] Hin. specialize (Hhs' k _ Hin).
      cbn [calls_handler_f3] in Hhs'.
      apply andb_true_iff in Hhs' as [Hhs' _].
      apply andb_true_iff in Hhs' as [Hhs' _].
      apply andb_true_iff in Hhs' as [Hlen _]. apply Nat.eqb_eq in Hlen.
      exact (handler_pdescs_maps false params md hbody Hlen). }
    assert (HD : ∀ k ps hb, (k, (ps, hb)) ∈ D ->
              forallb pdesc_ok ps = true).
    { intros k ps hb Hin. unfold D in Hin.
      apply in_map_iff in Hin as ([k' h] & Heq & Hin).
      cbn in Heq. injection Heq as <- <- <-.
      specialize (Hhs' k' h Hin). destruct h as [params md hbody].
      cbn [calls_handler_f3] in Hhs'.
      apply andb_true_iff in Hhs' as [Hhs' _].
      apply andb_true_iff in Hhs' as [_ Hok]. exact Hok. }
    pose proof (cinv_conts C en enu D Hci HD) as Hci'.
    assert (Hmk : map (fun '(k, (ps, _)) => (k, map pd_keep ps)) D
                  = map (fun '(k, h) => (k, handler_mask sol false h))
                      handlers).
    { unfold D. rewrite map_map. apply map_ext_in. intros [k h] Hin.
      cbn. rewrite (proj2 (Hmaps k h Hin)). reflexivity. }
    assert (Hdk : map (fun '(k, (ps, _)) => (k, map pd_decision ps)) D
                  = map (fun '(k, h) =>
                           (k, handler_decisions true sol unboxed false h))
                      handlers).
    { unfold D. rewrite map_map. apply map_ext_in. intros [k h] Hin.
      cbn. rewrite (proj1 (Hmaps k h Hin)). reflexivity. }
    assert (Hfst : map fst D = map fst handlers).
    { unfold D. rewrite map_map. apply map_ext. intros [k h].
      reflexivity. }
    set (en' := bind_continuations en
                  (map (fun '(k, h) => (k, handler_mask sol false h))
                     handlers)).
    set (enu' := ubind_continuations enu
                   (map (fun '(k, h) =>
                           (k, handler_decisions true sol unboxed false h))
                      handlers)).
    assert (Hci'' : cinv (D ++ C) en' enu').
    { unfold en', enu'. rewrite <- Hmk, <- Hdk. exact Hci'. }
    clear Hci'. rename Hci'' into Hci'.
    destruct (IH en' enu' (D ++ C) ctx
                (FlambdaReaperFlowProofs.rec_conts ce handlers) Hci')
      as (IHuc & IHfc & IHbs & IHincl).
    { exact Hbody. }
    { split; intros; [ apply Hvars | apply Hconts ]; apply in_or_app;
        right; assumption. }
    { exact Hbody_edges. }
    assert (Hhandlers :
      ∀ hs : list (continuation * continuation_handler),
        (∀ k h, (k, h) ∈ hs -> (k, h) ∈ handlers) ->
        let hps :=
          map (fun '(k, h) =>
                 (k, rebuild_handler_cc F sol cannot_change code_deps pdc
                       unit_code_ids codes ff en' false
                       (handler_mask sol false h) h))
            hs
        in
        let hus :=
          map (fun '(k, h) =>
                 (k, rebuild_handler_u true F sol cannot_change code_deps
                       pdc unit_code_ids codes unboxed ffu [] enu' false
                       (handler_decisions true sol unboxed false h) h))
            hs
        in
        uc_handlers unboxed dead CSf NRf RCHf OVf REMf (K_of (D ++ C)) hps hus
          (map (fun '(k, h) =>
                  (k, Kshape
                        (flat_map pshape_of
                           (handler_pdescs sol unboxed false h))
                        (flat_map kept_kind
                           (handler_pdescs sol unboxed false h))
                        true))
             hs)
        ∧ (handlers_bind_statics hus = true ->
           handlers_bind_statics hps = true)
        ∧ incl (flat_map (fun '(_, h) => free_continuations_of_handler h) hus)
            (flat_map (fun '(_, h) => free_continuations_of_handler h) hps)
        ∧ incl (flat_map (fun '(_, h) => free_continuations_of_handler h)
                  hus)
            (flat_map
               (fun '(_, h) => live_conts_handler F sol code_deps codes h)
               hs)
        ∧ map fst hps = map fst hus
        ∧ map fst hps = map fst hs).
    { induction hs as [ | [k h] hs IHhs]; intros Hsub; cbv zeta.
      - cbn [map flat_map]. split; [ apply uc_handlers_nil | ].
        split; [ intros H; exact H | split; [ intros ? [] | ] ].
        split; [ intros ? [] | split; reflexivity ].
      - destruct IHhs as (Hub_hs & Hbs_hs & Hfc_hs & Hincl_hs & Hfst_hs
                          & Hfst_hs').
        { intros k' h' Hin. apply Hsub. right. exact Hin. }
        assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
        destruct (IHh k h Hin en' enu' (D ++ C) ctx
                    (FlambdaReaperFlowProofs.rec_conts ce handlers) false
                    Hci')
          with (dead_check := false)
          as (ps_c & ps_u & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd'
              & Hsk' & Hkinds & _ & _ & Hhres).
        { exact (Hhs' k h Hin). }
        { split; intros; [ apply Hvars | apply Hconts ]; apply in_or_app;
            left; apply in_flat_map; exists (k, h);
            (split; [ exact Hin | ]); [ assumption | right; assumption ]. }
        { exact (every_in _ _ (k, h) Hhs_edges Hin). }
        cbn [map flat_map]. rewrite Hp, Hu.
        destruct Hhres as (Hhuc & Hhfc & Hhbs & Hhincl).
        split.
        { rewrite <- Hkinds. apply uc_handlers_cons; assumption. }
        unfold handlers_bind_statics in *. cbn [existsb map fst].
        cbn [free_continuations_of_handler handler_binds_statics].
        split; [ exact (orb_impl2 _ _ _ _ Hhbs Hbs_hs) | ].
        split; [ apply incl_app_app; [ exact Hhfc | exact Hfc_hs ] | ].
        split.
        { intros x Hx. apply in_app_or in Hx as [Hx | Hx]; apply in_or_app;
            [ left; exact (Hhincl x Hx) | right; exact (Hincl_hs x Hx) ]. }
        split; f_equal; assumption. }
    pose proof (Hhandlers handlers (fun k h Hin => Hin)) as Hh.
    cbv zeta in Hh.
    destruct Hh as (Hub_hs & Hbs_hs & Hfc_hs & Hincl_hs & Hfst_hs & Hfst_hs').
    rewrite cc_rebuild_let_cont_rec, u_rebuild_let_cont_rec. cbv zeta.
    fold en' enu'.
    set (S := map (fun '(k, h) =>
                     (k, Kshape
                           (flat_map pshape_of
                              (handler_pdescs sol unboxed false h))
                           (flat_map kept_kind
                              (handler_pdescs sol unboxed false h))
                           true))
                handlers) in *.
    assert (HK : K_of (D ++ C) = S ++ K_of C).
    { unfold K_of, D, S. rewrite map_app, map_map. f_equal.
      apply map_ext. intros [k h]. reflexivity. }
    rewrite HK in Hub_hs, IHuc.
    split; [ apply (uc_let_cont_rec unboxed dead CSf NRf RCHf OVf REMf
                      (K_of C) S);
             assumption | ].
    cbn [free_continuations_of_expr binds_statics live_conts].
    unfold handlers_bind_statics in Hbs_hs.
    split.
    { rewrite <- Hfst_hs. apply incl_without_mono, incl_app_app;
        [ exact Hfc_hs | exact IHfc ]. }
    split; [ exact (orb_impl2 _ _ _ _ Hbs_hs IHbs) | ].
    intros k Hk. apply in_without_iff in Hk as [Hk Hne].
    apply in_without_iff. split.
    + apply in_app_or in Hk as [Hk | Hk]; apply in_or_app;
        [ left; exact (Hincl_hs k Hk) | right; exact (IHincl k Hk) ].
    + intros H. apply Hne. rewrite <- Hfst_hs.
      exact (eq_ind_r (fun l => k ∈ l) H Hfst_hs').
  - (* Apply *)
    intros ap en enu C ctx ce Hci Hloc [Hvars Hconts] _.
    cbn [calls_expr_f3] in Hloc. cbn [variables_of continuations_of] in Hvars,
      Hconts.
    apply (call_rebuilt en enu C ap Hci); try assumption.
    + intros z Hz. exact (proj1 (Hvars z Hz)).
    + intros Heq. apply (proj1 (Hconts _ (or_intror (or_introl eq_refl)))).
      exact Heq.
    + intros Heq. apply (proj2 (Hconts _ (or_intror (or_introl eq_refl)))).
      exact Heq.
  - (* Apply_cont *)
    intros ac en enu C ctx ce Hci Hloc [Hvars _] _.
    unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
    cbn [calls_expr_f3] in Hloc. cbn [variables_of] in Hvars.
    unfold rebuild_apply_cont, rebuild_apply_cont_u.
    destruct (rewrite_apply_cont_rel2 en enu C ac Hci
                (fun z Hz => proj1 (Hvars z Hz)) Hloc)
      as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
      rewrite Hp, Hu; [ exact (uc_result_invalid _ _) | ].
    split; [ constructor; exact Hrel | ].
    destruct Hrel as (Hk & Htrap & _).
    cbn [free_continuations_of_expr binds_statics live_conts].
    unfold apply_cont_continuations. rewrite Hk, Htrap.
    split; [ intros x Hx; exact Hx | split; [ intros H; exact H | ] ].
    unfold rewrite_apply_cont in Hp. unfold live_apply_cont.
    destruct (existsb _ _); [ discriminate Hp | ]. injection Hp as <-.
    intros k Hk'. exact Hk'.
  - (* Switch *)
    intros scrutinee arms en enu C ctx ce Hci Hloc [Hvars _] _.
    cbn [calls_expr_f3] in Hloc. apply andb_true_iff in Hloc as [Hs Harms].
    cbn [variables_of] in Hvars.
    assert (Hfr0 : ∀ z, z ∈ variables_in_simple scrutinee
                          ++ flat_map (fun arm => variables_in_apply_cont
                                                    (snd arm)) arms ->
                        fresh z)
      by (intros z Hz; exact (proj1 (Hvars z Hz))).
    unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
    unfold rebuild_switch, rebuild_switch_u.
    assert (Hcompact : ∀ (d : Z) (ac : apply_cont),
               (d, ac) ∈ Keyed.compact Z.eqb arms ->
               (d, ac) ∈ arms).
    { intros d ac Hin.
      apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hin.
      apply (keyed_lookup_in Z.eqb z_eqb_decides_equality) in Hin.
      exact Hin. }
    assert (Harm_ok : ∀ (d : Z) (ac : apply_cont), (d, ac) ∈ arms ->
              apply_cont_ok2 sol unboxed (env_kinds en) C ac = true
              ∧ (∀ z, z ∈ variables_in_apply_cont ac -> fresh z)
              ∧ (existsb (simple_has_no_source sol) (ac_args ac) = false ->
                 incl (apply_cont_continuations ac)
                   (live_conts F sol code_deps codes
                      (Switch scrutinee arms)))).
    { intros d ac Hin. split; [ | split ].
      - rewrite forallb_forall in Harms. exact (Harms _ Hin).
      - intros z Hz. apply Hfr0. apply in_or_app. right.
        apply in_flat_map. exists (d, ac). split; [ exact Hin | exact Hz ].
      - intros Hns k Hk. cbn [live_conts]. apply in_flat_map.
        exists (d, ac). split; [ exact Hin | ]. cbn [snd].
        unfold live_apply_cont. rewrite Hns. exact Hk. }
    assert (Harms' :
      ∀ l : list (Z * apply_cont),
        (∀ (d : Z) (ac : apply_cont), (d, ac) ∈ l -> (d, ac) ∈ arms) ->
        Forall2
          (fun arm arm' : Z * apply_cont =>
             fst arm' = fst arm
             ∧ ac_rel2 unboxed dead (K_of C) (snd arm) (snd arm')
             ∧ incl (apply_cont_continuations (snd arm'))
                 (live_conts F sol code_deps codes (Switch scrutinee arms)))
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont sol en ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol unboxed enu ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)).
    { induction l as [ | [d ac] l IHl]; intros Hl; [ constructor | ].
      cbn [flat_map].
      destruct (Harm_ok d ac (Hl d ac (or_introl eq_refl)))
        as (Hok & Hfr & Hinc).
      assert (Hrest := IHl (fun d' ac' Hin => Hl d' ac' (or_intror Hin))).
      destruct (rewrite_apply_cont_rel2 en enu C ac Hci Hfr Hok)
        as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
        rewrite Hp, Hu; [ exact Hrest | ].
      constructor; [ | exact Hrest ].
      split; [ reflexivity | split; [ exact Hrel | ] ].
      cbn [snd]. rewrite (ac_rel2_conts _ _ _ Hrel).
      assert (Hns : existsb (simple_has_no_source sol) (ac_args ac) = false).
      { unfold rewrite_apply_cont in Hp.
        destruct (existsb _ _); [ discriminate Hp | reflexivity ]. }
      unfold rewrite_apply_cont in Hp. rewrite Hns in Hp.
      injection Hp as <-. exact (Hinc Hns). }
    pose proof (Harms' _ Hcompact) as Hrel.
    rewrite (rewrite_simple_kinds sol_u (ue_env enu) en scrutinee
               (ci_kinds _ _ _ Hci)).
    assert (Hs' : plain_simple unboxed scrutinee = true).
    { apply (strong_simple2 scrutinee Hs). intros z Hz.
      apply Hfr0. apply in_or_app. left. exact Hz. }
    rewrite (rewrite_simple_u sol unboxed en scrutinee Hs').
    destruct Hrel as [ | arm arm' l l' Harm Hrel ];
      [ exact (uc_result_invalid _ _) | ].
    cbn [Keyed.is_empty].
    split.
    { apply uc_switch; [ apply rewrite_plain2;
                         [ exact (ci_kinds_ok _ _ _ Hci) | exact Hs' ] | ].
      constructor; [ destruct Harm as (Ha & Hb & _); split; assumption | ].
      clear -Hrel. induction Hrel as [ | a a' l l' (Ha & Hb & _) _ IH ];
        constructor; [ split; assumption | exact IH ]. }
    split; [ | split; [ intros H; exact H | ] ].
    + apply eq_incl_sym. cbn [free_continuations_of_expr flat_map].
      destruct Harm as (_ & Hac & _).
      f_equal; [ symmetry; exact (ac_rel2_conts _ _ _ Hac) | ].
      clear -Hrel. induction Hrel as [ | a a' l l' (_ & Hac & _) _ IH ];
        [ reflexivity | ].
      cbn [flat_map]. f_equal; [ symmetry; exact (ac_rel2_conts _ _ _ Hac)
                               | exact IH ].
    + intros k Hk. cbn [free_continuations_of_expr flat_map] in Hk.
      apply in_app_or in Hk as [Hk | Hk];
        [ destruct Harm as (_ & _ & Hinc); exact (Hinc k Hk) | ].
      clear -Hrel Hk. induction Hrel as [ | a a' l l' (_ & _ & Hinc) _ IH ];
        [ destruct Hk | ].
      cbn [flat_map] in Hk. apply in_app_or in Hk as [Hk | Hk];
        [ exact (Hinc k Hk) | exact (IH Hk) ].
  - (* Invalid *)
    intros en enu C ctx ce _ _ _ _. exact (uc_result_invalid _ _).
  - (* Handler *)
    intros params metadata handler IH en enu C ctx ce is_exn Hci
      Hloc [Hvars Hconts] Hedges dead_check.
    cbn [calls_handler_f3] in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hloc Hps_ok].
    apply andb_true_iff in Hloc as [Hlen Hkinds_ok]. apply Nat.eqb_eq in Hlen.
    cbn [variables_in_handler continuations_in_handler] in Hvars, Hconts.
    pose proof (with_kinds_combine params (hm_params_kinds metadata) Hlen)
      as Hbound.
    set (bound := with_kinds params (hm_params_kinds metadata)) in *.
    set (mask := handler_mask sol is_exn (Handler params metadata handler)).
    assert (Hmlen : length mask = length bound)
      by (unfold mask, handler_mask; apply params_to_keep_length).
    assert (Hfst : map fst bound = params)
      by (rewrite Hbound; apply map_fst_combine, Hlen).
    assert (Hsnd : map snd bound = hm_params_kinds metadata)
      by (rewrite Hbound; apply map_snd_combine, Hlen).
    assert (Hbk : ∀ x kind, (x, kind) ∈ bound ->
              var_ok x ∧ kind_ok kind = true).
    { intros x kind Hin. split.
      - apply Hvars. apply in_or_app. left. rewrite <- Hfst.
        apply (in_map fst) in Hin. exact Hin.
      - rewrite forallb_forall in Hkinds_ok. apply Hkinds_ok.
        rewrite <- Hsnd. apply (in_map snd) in Hin. exact Hin. }
    assert (Hds : handler_decisions true sol unboxed is_exn
                    (Handler params metadata handler)
                  = map decide_param (combine bound mask))
      by reflexivity.
    assert (Hps : handler_pdescs sol unboxed is_exn
                    (Handler params metadata handler)
                  = map (fun '(d, keep, kind) => Pdesc d keep kind)
                      (combine (combine (map decide_param (combine bound mask))
                                  mask)
                         (map snd bound))).
    { unfold handler_pdescs. rewrite Hds, Hsnd. reflexivity. }
    rewrite Hps in Hps_ok |- *. rewrite Hds.
    destruct (params_rel2_gen bound mask Hmlen Hbk Hps_ok)
      as (Hrel & Hdecs & Hkeeps & Hkinds).
    assert (Hci' : cinv C (bind_variables en bound)
                     (ubind_variables enu bound))
      by (apply cinv_bind; [ exact Hci | intros y k Hin;
                             exact (proj2 (Hbk y k Hin)) ]).
    destruct (IH (bind_variables en bound) (ubind_variables enu bound) C
                ctx ce Hci')
      as (IHuc & IHfc & IHbs & IHincl).
    { cbn [ue_env bind_variables bind_variables_then env_kinds].
      rewrite Hbound. exact Hbody. }
    { split; intros; [ apply Hvars, in_or_app; right | apply Hconts ];
        assumption. }
    { exact Hedges. }
    rewrite cc_rebuild_handler, u_rebuild_handler. cbv zeta. fold bound.
    exists (filter_kept mask bound),
      (params_of_decisions bound (map decide_param (combine bound mask))),
      (with_params_kinds metadata
         (filter_kept mask (hm_params_kinds metadata))),
      (with_params_kinds metadata
         (map snd
            (params_of_decisions bound
               (map decide_param (combine bound mask))))).
    eexists _, _.
    split; [ rewrite <- filter_kept_map, Hfst; reflexivity | ].
    split; [ reflexivity | ].
    split; [ exact Hrel | ].
    split; [ rewrite hm_with_params_kinds, <- Hsnd, filter_kept_map;
             reflexivity | ].
    split; [ apply hm_with_params_kinds | ].
    split; [ apply with_params_kinds_subkinds | ].
    split; [ exact Hkinds | ].
    split; [ exact Hdecs | ].
    split; [ exact Hkeeps | ].
    destruct (dead_check && _); [ exact (uc_result_invalid _ _) | ].
    split; [ exact IHuc | split; [ exact IHfc | split; [ exact IHbs | ] ] ].
    exact IHincl.
Qed.

End RebuildsCalls.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* With [check_unboxed_uses], the closures among the usages of a name
   unboxed add nothing to them. *)
Lemma add_usages_unboxed (F : facts) (T : tables)
    (cm : finite_map variable) (unboxed : list (node * unboxed_fields))
    (n : node) (fields : unboxed_fields) (follow : bool) :
  check_unboxed_uses F T unboxed = true -> (n, fields) ∈ unboxed ->
  add_usages_through_function_slots F T cm follow (get_direct_usages T [n])
  = get_direct_usages T [n].
Proof.
  intros Huses Hn.
  assert (Hblock : ∀ x, x ∈ get_direct_usages T [n] ->
            ∀ y f base, (y, f, base) ∈ f_accessor F ->
              node_eqb base x = true -> is_block_like_field f = true).
  { intros x Hx y f base Hacc Hb. apply node_eqb_eq in Hb. subst base.
    apply get_direct_usages_in in Hx as [Hpair _].
    unfold check_unboxed_uses in Huses. rewrite forallb_forall in Huses.
    specialize (Huses _ Hacc). cbn beta iota in Huses.
    apply orb_true_iff in Huses as [Hf | Hall]; [ exact Hf | ].
    rewrite forallb_forall in Hall. specialize (Hall _ Hn).
    cbn beta iota in Hall. rewrite Hpair in Hall. discriminate Hall. }
  generalize dependent (get_direct_usages T [n]). intros out Hblock.
  assert (Hnone : usages_through_function_slots F T cm follow out = []).
  { destruct (usages_through_function_slots F T cm follow out)
      as [ | z rest] eqn:E; [ reflexivity | exfalso ].
    assert (Hz : z ∈ usages_through_function_slots F T cm follow out)
      by (rewrite E; left; reflexivity).
    clear E. unfold usages_through_function_slots in Hz.
    apply in_flat_map in Hz as (x & Hx & Hz).
    apply in_app_or in Hz as [Hz | Hz];
      [ destruct follow; [ | destruct Hz ] | ];
      apply in_flat_map in Hz as ([[y f] base] & Hacc & Hz);
      cbn beta iota in Hz;
      match type of Hz with
      | _ ∈ (if ?c then _ else _) => destruct c eqn:Ht; [ | destruct Hz ]
      end;
      apply andb_true_iff in Ht as [Hb Hf];
      pose proof (Hblock x Hx y f base Hacc Hb) as Hbl;
      destruct f; try discriminate Hbl; discriminate Hf. }
  unfold add_usages_through_function_slots. simpl. rewrite Hnone.
  reflexivity.
Qed.

(* The fragment of FlambdaReaperUnboxingRefinesCallsSpec.v is in that of
   FlambdaReaperUnboxingRefinesOverSpec.v. *)
Section InOver.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.

Lemma ind_code_in_over (kind : call_kind) (kps : list pdesc) (RA : arity)
    (id : code_id) (md : code_metadata) :
  ind_code_ok sol cannot_change code_deps unit_code_ids unboxed kps RA id md
  = true ->
  ind_code_ok_o sol cannot_change code_deps unit_code_ids unboxed kind kps RA
    id md
  = true.
Proof.
  unfold ind_code_ok, ind_code_ok_o.
  destruct (code_change_u _ _ _ _ _ _ _); [ | discriminate ].
  destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ]; [ | discriminate ].
  destruct (code_exn_decision _ _ _ _); [ | discriminate ].
  intros H. apply andb_true_iff in H as [H1 H2].
  rewrite H1. cbn [andb].
  destruct (known_arity_call kind); [ reflexivity | exact H2 ].
Qed.

Lemma ind_call_in_over (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  ind_call_ok F sol cannot_change code_deps unit_code_ids codes unboxed kinds
    C ap f
  = true ->
  ind_call_ok_o F sol cannot_change code_deps unit_code_ids codes unboxed
    kinds C ap f
  = true.
Proof.
  unfold ind_call_ok, ind_call_ok_o. intros H.
  apply andb_true_iff in H as [Hpre Hm]. rewrite Hpre. cbn [andb].
  destruct (lookup (ap_continuation ap) C) as [[kps [ | ]] | ];
    try discriminate Hm.
  apply andb_true_iff in Hm as [Hm Hall]. rewrite Hm. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  exact (ind_code_in_over _ _ _ id md Hall).
Qed.

Lemma call_in_over (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  call_ok F sol cannot_change code_deps unit_code_ids codes unboxed nr kinds C
    ap
  = true ->
  call_ok_o F sol cannot_change code_deps unit_code_ids codes unboxed nr kinds
    C ap
  = true.
Proof.
  unfold call_ok, call_ok_o.
  destruct (ap_callee ap) as [f | sym | cst];
    [ | exact (fun H => H) | exact (fun H => H) ].
  destruct (code_id_actually_called F sol ap) as [cid | ];
    [ exact (fun H => H) | ].
  apply ind_call_in_over.
Qed.

Lemma calls_in_over :
  (∀ e : expr, ∀ kinds C,
     calls_expr F sol cannot_change code_deps unit_code_ids codes unboxed nr
       kinds C e
     = true ->
     calls_expr_o F sol cannot_change code_deps unit_code_ids codes unboxed
       nr kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       calls_handler F sol cannot_change code_deps unit_code_ids codes
         unboxed nr kinds C ps h
       = true ->
       calls_handler_o F sol cannot_change code_deps unit_code_ids codes
         unboxed nr kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [calls_expr calls_handler calls_expr_o calls_handler_o].
  - intros x kind d body IH kinds C H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
  - intros bindings body IHc IH kinds C H.
    apply andb_true_iff in H as [Hbs H]. rewrite (IH _ _ H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding calls_binding_o calls_code calls_code_o code_body]
      in Hbs, IHb |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hb).
  - intros k h body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ exact (IHh _ _ _ Hh) | exact (IH _ _ H) ].
  - intros handlers body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ Hh).
  - intros ap kinds C H. exact (call_in_over kinds C ap H).
  - intros ac kinds C H. exact H.
  - intros scrutinee arms kinds C H. exact H.
  - intros kinds C H. exact H.
  - intros params metadata handler IH kinds C ps H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
Qed.

End InOver.

(* The fragment of FlambdaReaperUnboxingRefinesOverSpec.v is in that of
   FlambdaReaperUnboxingRefinesOver1Spec.v. *)
Section InOver1.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.

Lemma ind_code_in_over1 (ap : apply_expr) (kps : list pdesc) (RA : arity)
    (id : code_id) (md : code_metadata) :
  ind_code_ok_o sol cannot_change code_deps unit_code_ids unboxed
    (ap_call_kind ap) kps RA id md
  = true ->
  ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
    (never_over codes ap) kps RA id md
  = true.
Proof.
  unfold ind_code_ok_o, ind_code_ok_o1.
  destruct (code_change_u _ _ _ _ _ _ _); [ | discriminate ].
  destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ]; [ | discriminate ].
  destruct (code_exn_decision _ _ _ _); [ | discriminate ].
  intros H. apply andb_true_iff in H as [H1 H2].
  rewrite H1. cbn [andb]. unfold never_over.
  destruct (known_arity_call (ap_call_kind ap)); [ reflexivity | ].
  cbn [orb] in H2.
  apply orb_true_iff in H2 as [H2 | H2]; rewrite H2; rewrite ?orb_true_r;
    reflexivity.
Qed.

Lemma ind_call_in_over1 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  ind_call_ok_o F sol cannot_change code_deps unit_code_ids codes unboxed
    kinds C ap f
  = true ->
  ind_call_ok_o1 F sol cannot_change code_deps unit_code_ids codes unboxed
    kinds C ap f
  = true.
Proof.
  unfold ind_call_ok_o, ind_call_ok_o1. intros H.
  apply andb_true_iff in H as [Hpre Hm]. rewrite Hpre. cbn [andb].
  destruct (lookup (ap_continuation ap) C) as [[kps [ | ]] | ];
    try discriminate Hm.
  apply andb_true_iff in Hm as [Hm Hall]. rewrite Hm. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  exact (ind_code_in_over1 ap _ _ id md Hall).
Qed.

Lemma call_in_over1 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  call_ok_o F sol cannot_change code_deps unit_code_ids codes unboxed nr
    kinds C ap
  = true ->
  call_ok_o1 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    kinds C ap
  = true.
Proof.
  unfold call_ok_o, call_ok_o1.
  destruct (ap_callee ap) as [f | sym | cst];
    [ | exact (fun H => H) | exact (fun H => H) ].
  destruct (code_id_actually_called F sol ap) as [cid | ];
    [ exact (fun H => H) | ].
  apply ind_call_in_over1.
Qed.

Lemma calls_in_over1 :
  (∀ e : expr, ∀ kinds C,
     calls_expr_o F sol cannot_change code_deps unit_code_ids codes unboxed
       nr kinds C e
     = true ->
     calls_expr_o1 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       calls_handler_o F sol cannot_change code_deps unit_code_ids codes
         unboxed nr kinds C ps h
       = true ->
       calls_handler_o1 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [calls_expr_o calls_handler_o calls_expr_o1 calls_handler_o1].
  - intros x kind d body IH kinds C H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
  - intros bindings body IHc IH kinds C H.
    apply andb_true_iff in H as [Hbs H]. rewrite (IH _ _ H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding_o calls_binding_o1 calls_code_o calls_code_o1
         code_body]
      in Hbs, IHb |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hb).
  - intros k h body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ exact (IHh _ _ _ Hh) | exact (IH _ _ H) ].
  - intros handlers body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ Hh).
  - intros ap kinds C H. exact (call_in_over1 kinds C ap H).
  - intros ac kinds C H. exact H.
  - intros scrutinee arms kinds C H. exact H.
  - intros kinds C H. exact H.
  - intros params metadata handler IH kinds C ps H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
Qed.

End InOver1.

Section InFlow.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.
Variable reach : variable -> code_id -> bool.

Lemma ind_call_in_flow (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  ind_call_ok_o1 F sol cannot_change code_deps unit_code_ids codes unboxed
    kinds C ap f
  = true ->
  ind_call_ok_f F sol cannot_change code_deps unit_code_ids codes unboxed
    reach kinds C ap f
  = true.
Proof.
  unfold ind_call_ok_o1, ind_call_ok_f. intros H.
  apply andb_true_iff in H as [Hpre Hm]. rewrite Hpre. cbn [andb].
  destruct (lookup (ap_continuation ap) C) as [[kps [ | ]] | ];
    try discriminate Hm.
  apply andb_true_iff in Hm as [Hm Hall]. rewrite Hm. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  rewrite Hall, orb_true_r. reflexivity.
Qed.

Lemma call_in_flow (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  call_ok_o1 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    kinds C ap
  = true ->
  call_ok_f F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach kinds C ap
  = true.
Proof.
  unfold call_ok_o1, call_ok_f.
  destruct (ap_callee ap) as [f | sym | cst];
    [ | exact (fun H => H) | exact (fun H => H) ].
  destruct (code_id_actually_called F sol ap) as [cid | ];
    [ exact (fun H => H) | ].
  apply ind_call_in_flow.
Qed.

Lemma calls_in_flow :
  (∀ e : expr, ∀ kinds C,
     calls_expr_o1 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr kinds C e
     = true ->
     calls_expr_f F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       calls_handler_o1 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr kinds C ps h
       = true ->
       calls_handler_f F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [calls_expr_o1 calls_handler_o1 calls_expr_f calls_handler_f].
  - intros x kind d body IH kinds C H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
  - intros bindings body IHc IH kinds C H.
    apply andb_true_iff in H as [Hbs H]. rewrite (IH _ _ H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding_o1 calls_binding_f calls_code_o1 calls_code_f
         code_body]
      in Hbs, IHb |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hb).
  - intros k h body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ exact (IHh _ _ _ Hh) | exact (IH _ _ H) ].
  - intros handlers body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ Hh).
  - intros ap kinds C H. exact (call_in_flow kinds C ap H).
  - intros ac kinds C H. exact H.
  - intros scrutinee arms kinds C H. exact H.
  - intros kinds C H. exact H.
  - intros params metadata handler IH kinds C ps H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
Qed.

End InFlow.

(* The reach of the second fragment that holds the first: that of the
   first for a call of known arity, and every code otherwise. *)
Definition reach2 (reach : variable -> code_id -> bool) (known : bool)
    : variable -> code_id -> bool :=
  if known then reach else fun _ _ => true.

Section InFlow2.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.
Variable reach : variable -> code_id -> bool.
(* A reach of the second fragment that is that of the first for a call of
   known arity. *)
Variable reach' : bool -> variable -> code_id -> bool.
Hypothesis reach'_known : ∀ (f : variable) (id : code_id),
  reach' true f id = reach f id.

Lemma ind_call_in_flow2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  ind_call_ok_f F sol cannot_change code_deps unit_code_ids codes unboxed
    reach kinds C ap f
  = true ->
  ind_call_ok_f2 F sol cannot_change code_deps unit_code_ids codes unboxed
    reach' kinds C ap f
  = true.
Proof.
  unfold ind_call_ok_f, ind_call_ok_f2. intros H.
  apply andb_true_iff in H as [Hpre Hm]. rewrite Hpre. cbn [andb].
  destruct (lookup (ap_continuation ap) C) as [[kps [ | ]] | ];
    try discriminate Hm.
  apply andb_true_iff in Hm as [Hm Hall]. rewrite Hm. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  apply orb_true_iff in Hall as [Hex | Hok];
    [ | rewrite Hok, orb_true_r; reflexivity ].
  apply andb_true_iff in Hex as [Hk Hr]. rewrite Hk, reach'_known, Hr.
  unfold never_over. rewrite Hk. reflexivity.
Qed.

Lemma call_in_flow2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  call_ok_f F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach kinds C ap
  = true ->
  call_ok_f2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach' kinds C ap
  = true.
Proof.
  unfold call_ok_f, call_ok_f2.
  destruct (ap_callee ap) as [f | sym | cst];
    [ | exact (fun H => H) | exact (fun H => H) ].
  destruct (code_id_actually_called F sol ap) as [cid | ];
    [ exact (fun H => H) | ].
  apply ind_call_in_flow2.
Qed.

Lemma calls_in_flow2 :
  (∀ e : expr, ∀ kinds C,
     calls_expr_f F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach kinds C e
     = true ->
     calls_expr_f2 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach' kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       calls_handler_f F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach kinds C ps h
       = true ->
       calls_handler_f2 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach' kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [calls_expr_f calls_handler_f calls_expr_f2 calls_handler_f2].
  - intros x kind d body IH kinds C H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
  - intros bindings body IHc IH kinds C H.
    apply andb_true_iff in H as [Hbs H]. rewrite (IH _ _ H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding_f calls_binding_f2 calls_code_f calls_code_f2
         code_body]
      in Hbs, IHb |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hb).
  - intros k h body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ exact (IHh _ _ _ Hh) | exact (IH _ _ H) ].
  - intros handlers body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ Hh).
  - intros ap kinds C H. exact (call_in_flow2 kinds C ap H).
  - intros ac kinds C H. exact H.
  - intros scrutinee arms kinds C H. exact H.
  - intros kinds C H. exact H.
  - intros params metadata handler IH kinds C ps H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
Qed.

End InFlow2.


(* The fragment of FlambdaReaperUnboxingRefinesReach2Spec.v is in that of
   FlambdaReaperUnboxingRefinesOver2Spec.v, whatever the code [ovr]. *)
Section InOver2.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.
Variable reach : bool -> variable -> code_id -> bool.
Variable ovr : code_id -> bool.

Lemma ind_code_ok_o1_or (no no' : bool) (kps : list pdesc) (RA : arity)
    (id : code_id) (md : code_metadata) :
  ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed no kps RA
    id md
  = true ->
  ind_code_ok_o1 sol cannot_change code_deps unit_code_ids unboxed
    (no || no') kps RA id md
  = true.
Proof.
  intros H. destruct no; [ exact H | ]. destruct no'; [ | exact H ].
  revert H. unfold ind_code_ok_o1.
  destruct (code_change_u _ _ _ _ _ _ _); [ | discriminate ].
  destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ]; [ | discriminate ].
  destruct (code_exn_decision _ _ _ _); [ | discriminate ].
  intros H. apply andb_true_iff in H as [H1 _]. rewrite H1. reflexivity.
Qed.

Lemma ind_call_in_over2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  ind_call_ok_f2 F sol cannot_change code_deps unit_code_ids codes unboxed
    reach kinds C ap f
  = true ->
  ind_call_ok_o2 F sol cannot_change code_deps unit_code_ids codes unboxed
    reach ovr kinds C ap f
  = true.
Proof.
  unfold ind_call_ok_f2, ind_call_ok_o2. intros H.
  apply andb_true_iff in H as [Hpre Hm]. rewrite Hpre. cbn [andb].
  destruct (lookup (ap_continuation ap) C) as [[kps [ | ]] | ];
    try discriminate Hm.
  apply andb_true_iff in Hm as [Hm Hall]. rewrite Hm. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  apply orb_true_iff in Hall as [Hex | Hok];
    [ rewrite Hex; reflexivity | ].
  rewrite (ind_code_ok_o1_or _ (negb (ovr id)) _ _ _ _ Hok), orb_true_r.
  reflexivity.
Qed.

Lemma call_in_over2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  call_ok_f2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach kinds C ap
  = true ->
  call_ok_o2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach ovr kinds C ap
  = true.
Proof.
  unfold call_ok_f2, call_ok_o2.
  destruct (ap_callee ap) as [f | sym | cst];
    [ | exact (fun H => H) | exact (fun H => H) ].
  destruct (code_id_actually_called F sol ap) as [cid | ];
    [ exact (fun H => H) | ].
  apply ind_call_in_over2.
Qed.

Lemma calls_in_over2 :
  (∀ e : expr, ∀ kinds C,
     calls_expr_f2 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach kinds C e
     = true ->
     calls_expr_o2 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach ovr kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       calls_handler_f2 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach kinds C ps h
       = true ->
       calls_handler_o2 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach ovr kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [calls_expr_f2 calls_handler_f2 calls_expr_o2 calls_handler_o2].
  - intros x kind d body IH kinds C H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
  - intros bindings body IHc IH kinds C H.
    apply andb_true_iff in H as [Hbs H]. rewrite (IH _ _ H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding_f2 calls_binding_o2 calls_code_f2 calls_code_o2
         code_body]
      in Hbs, IHb |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hb).
  - intros k h body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ exact (IHh _ _ _ Hh) | exact (IH _ _ H) ].
  - intros handlers body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ Hh).
  - intros ap kinds C H. exact (call_in_over2 kinds C ap H).
  - intros ac kinds C H. exact H.
  - intros scrutinee arms kinds C H. exact H.
  - intros kinds C H. exact H.
  - intros params metadata handler IH kinds C ps H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
Qed.

End InOver2.

Lemma calls_unboxing_in_over2_reach (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : bool -> variable -> code_id -> bool)
    (ovr : code_id -> bool) (e : expr) :
  calls_unboxing_reach2_nr nr F T cannot_change code_deps dominated to_unbox
    reach e
  = true ->
  calls_unboxing_reach_over_nr nr F T cannot_change code_deps dominated
    to_unbox reach ovr e
  = true.
Proof.
  intros H. unfold calls_unboxing_reach2_nr in H.
  unfold calls_unboxing_reach_over_nr.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  exact (proj1 (calls_in_over2 _ _ _ _ _ _ _ nr reach ovr) e [] [] H1).
Qed.

(* The fragment of FlambdaReaperUnboxingRefinesOver2Spec.v is in that of
   FlambdaReaperUnboxingRefinesReach3Spec.v, whatever the code [rem], for a
   reach [reach'] narrower than [reach]. *)
Section InOver3.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.
Variable reach reach' : bool -> variable -> code_id -> bool.
Hypothesis reach'_le : ∀ (b : bool) (f : variable) (id : code_id),
  reach' b f id = true -> reach b f id = true.
Variable ovr rem : code_id -> bool.

Lemma ind_call_in_over3 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  ind_call_ok_o2 F sol cannot_change code_deps unit_code_ids codes unboxed
    reach ovr kinds C ap f
  = true ->
  ind_call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
    reach' ovr rem kinds C ap f
  = true.
Proof.
  unfold ind_call_ok_o2, ind_call_ok_f3. intros H.
  apply andb_true_iff in H as [Hpre Hm]. rewrite Hpre. cbn [andb].
  destruct (lookup (ap_continuation ap) C) as [[kps [ | ]] | ];
    try discriminate Hm.
  apply andb_true_iff in Hm as [Hm Hall]. rewrite Hm. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  apply orb_true_iff in Hall as [Hex | Hok];
    [ | rewrite Hok, orb_true_r; reflexivity ].
  apply andb_true_iff in Hex as [Hno Hr]. rewrite Hno. cbn [andb orb negb].
  destruct (reach' (known_arity_call (ap_call_kind ap)) f id) eqn:Hr';
    [ | reflexivity ].
  rewrite (reach'_le _ _ _ Hr') in Hr. discriminate Hr.
Qed.

Lemma call_in_over3 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  call_ok_o2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach ovr kinds C ap
  = true ->
  call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach' ovr rem kinds C ap
  = true.
Proof.
  unfold call_ok_o2, call_ok_f3.
  destruct (ap_callee ap) as [f | sym | cst];
    [ | exact (fun H => H) | exact (fun H => H) ].
  destruct (code_id_actually_called F sol ap) as [cid | ];
    [ exact (fun H => H) | ].
  apply ind_call_in_over3.
Qed.

Lemma calls_in_over3 :
  (∀ e : expr, ∀ kinds C,
     calls_expr_o2 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach ovr kinds C e
     = true ->
     calls_expr_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach' ovr rem kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       calls_handler_o2 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach ovr kinds C ps h
       = true ->
       calls_handler_f3 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach' ovr rem kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [calls_expr_o2 calls_handler_o2 calls_expr_f3 calls_handler_f3].
  - intros x kind d body IH kinds C H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
  - intros bindings body IHc IH kinds C H.
    apply andb_true_iff in H as [Hbs H]. rewrite (IH _ _ H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding_o2 calls_binding_f3 calls_code_o2 calls_code_f3
         code_body]
      in Hbs, IHb |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hb).
  - intros k h body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ exact (IHh _ _ _ Hh) | exact (IH _ _ H) ].
  - intros handlers body IHh IH kinds C H. cbv zeta in H |- *.
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ Hh).
  - intros ap kinds C H. exact (call_in_over3 kinds C ap H).
  - intros ac kinds C H. exact H.
  - intros scrutinee arms kinds C H. exact H.
  - intros kinds C H. exact H.
  - intros params metadata handler IH kinds C ps H.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2).
Qed.

End InOver3.

Lemma calls_unboxing_in_over3_reach (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node)
    (reach reach' : bool -> variable -> code_id -> bool)
    (ovr rem : code_id -> bool) (e : expr) :
  (∀ (b : bool) (f : variable) (id : code_id),
     reach' b f id = true -> reach b f id = true) ->
  calls_unboxing_reach_over_nr nr F T cannot_change code_deps dominated
    to_unbox reach ovr e
  = true ->
  calls_unboxing_reach_rem_nr nr F T cannot_change code_deps dominated
    to_unbox reach' ovr rem e
  = true.
Proof.
  intros Hle H. unfold calls_unboxing_reach_over_nr in H.
  unfold calls_unboxing_reach_rem_nr.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  exact (proj1 (calls_in_over3 _ _ _ _ _ _ _ nr reach reach' Hle ovr rem) e
           [] [] H1).
Qed.

Lemma calls_unboxing_in_flow2_reach (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : variable -> code_id -> bool)
    (reach' : bool -> variable -> code_id -> bool) (e : expr) :
  (∀ (f : variable) (id : code_id), reach' true f id = reach f id) ->
  calls_unboxing_reach_nr nr F T cannot_change code_deps dominated to_unbox
    reach e
  = true ->
  calls_unboxing_reach2_nr nr F T cannot_change code_deps dominated to_unbox
    reach' e
  = true.
Proof.
  intros Hk H. unfold calls_unboxing_reach_nr in H.
  unfold calls_unboxing_reach2_nr.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  exact (proj1 (calls_in_flow2 _ _ _ _ _ _ _ nr reach reach' Hk) e [] [] H1).
Qed.

Lemma calls_unboxing_in_flow_reach (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : variable -> code_id -> bool) (e : expr) :
  calls_unboxing_over1_nr nr F T cannot_change code_deps dominated to_unbox e
  = true ->
  calls_unboxing_reach_nr nr F T cannot_change code_deps dominated to_unbox
    reach e
  = true.
Proof.
  intros H. unfold calls_unboxing_over1_nr in H.
  unfold calls_unboxing_reach_nr.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  exact (proj1 (calls_in_flow _ _ _ _ _ _ _ nr reach) e [] [] H1).
Qed.


Lemma calls_unboxing_in_over1_nr :
  calls_unboxing_over_in_over1_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox e H.
  unfold calls_unboxing_over_nr in H. unfold calls_unboxing_over1_nr.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  exact (proj1 (calls_in_over1 _ _ _ _ _ _ _ nr) e [] [] H1).
Qed.

Lemma calls_unboxing_in_over_nr : calls_unboxing_in_over_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox e H.
  unfold calls_unboxing_nr in H. unfold calls_unboxing_over_nr.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  exact (proj1 (calls_in_over _ _ _ _ _ _ _ nr) e [] [] H1).
Qed.

(* The statement of the fix, for the fragment of
   FlambdaReaperUnboxingRefinesReach3Spec.v, where the flow of the first run
   is as [reach] says, it over-applies only code of [ovr], and what is left
   of an over-application calls only code of [rem]. *)
Lemma rebuild_unbox_modes_refines_cc_over3
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) (reach : bool -> variable -> code_id -> bool)
    (ovr rem : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_reach_rem_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox reach ovr rem e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  flow_run (fun b f id => reach b f id = true)
    (State
       (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
          (unit_code_deps flag u e) pdc e)
       [] [] empty_static_env empty_heap no_reads) ->
  over_run (fun id => ovr id = true)
    (State
       (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
          (unit_code_deps flag u e) pdc e)
       [] [] empty_static_env empty_heap no_reads) ->
  rem_run (fun id => rem id = true)
    (State
       (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
          (unit_code_deps flag u e) pdc e)
       [] [] empty_static_env empty_heap no_reads) ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode true (fun _ => false)
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox [] pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hcalls Hcheck Hflow Hover Hrem.
  set (F := traverse_unit flag u e) in *.
  unfold calls_unboxing_reach_rem_nr in Hcalls.
  apply andb_true_iff in Hcalls as [Hcalls Huses].
  pose proof (unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [Hdistinct Hfresh].
  pose proof (unboxed_fields_of_names_fresh F T (bound_code_my_closures e)
                dominated to_unbox
                (FlambdaNameAbstractionSpec.fresh_name (names_of e)))
    as Hnext.
  cbv zeta in Hnext. destruct Hnext as (Hnext & _ & _).
  unfold rebuild_unit_unbox_repr_mode, rebuild_unit_cc_from_tables,
    rebuild_unit_cc in Hflow, Hover, Hrem |- *.
  destruct (unit_unboxed_fields F T dominated to_unbox e) as [unboxed next]
    eqn:Hunboxed.
  cbn [fst] in Hcalls, Huses, Hdistinct, Hfresh.
  assert (Hnext' : (FlambdaNameAbstractionSpec.fresh_name (names_of e)
                    <= next)%positive).
  { unfold unit_unboxed_fields in Hunboxed. rewrite Hunboxed in Hnext.
    exact Hnext. }
  clear Hnext.
  assert (Hchanged :
    unit_changed_representations (fun _ => false) F T dominated to_unbox [] e
    = []) by reflexivity.
  rewrite Hchanged.
  pose proof (check_closed_sound is_local F T Hclosed) as Hcl.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts flag u e True
              Hfragment (or_introl I))
    as (noted & deps & cds & Hedges & _).
  assert (Hfields_read :
    ∀ (x : variable) (fields : unboxed_fields) (f : field)
      (v : unboxed_field),
      fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
      ∃ w base : node,
        g_accessor (graph_of_facts F) w f base
        ∧ s_usages (solution_of_tables T) (Node_variable x) base = true
        ∧ s_has_usage (solution_of_tables T) w = true).
  { intros x fields f v Hx Hin. apply (fields_of_in unboxed) in Hx.
    pose proof Hx as Hxu.
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hx' : (Node_variable x, fields)
                  ∈ fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                           dominated to_unbox
                           (FlambdaNameAbstractionSpec.fresh_name
                              (names_of e))))
      by (rewrite Hunboxed; exact Hx).
    destruct (unboxed_fields_of_names_in F T _ dominated to_unbox _ _ _ Hx')
      as (a & next' & ->).
    destruct (mk_unboxed_fields_in F T _ dominated to_unbox _ a _ next' f _
                Hin)
      as (use & Huse).
    destruct (get_fields_in F T _ f use Huse)
      as (w & base & Hacc & Hbase & Hw).
    rewrite (add_usages_unboxed F T _ unboxed _ _ true Huses Hxu) in Hbase.
    apply mem_node_In in Hbase.
    destruct (get_direct_usages_in T _ _ Hbase) as [Hus _].
    exists w, base. split; [ exact Hacc | split; [ exact Hus | ] ].
    destruct Hw as [Hw | Hw]; [ | exact Hw ].
    exact (has_usage_any _ _ _ Hcl _ Hw). }
  assert (Hreads_found :
    ∀ (x y : variable) (kind : flambda_kind) (f : field)
      (fields : unboxed_fields),
      fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
      g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
      is_block_like_field f = true ->
      is_dead_var (solution_of_tables T) x kind = false ->
      is_var_used (solution_of_tables T) x kind = true ->
      kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
      ∃ v : variable,
        Keyed.lookup field_eqb f fields = Some (Not_unboxed v)).
  { intros x y kind f fields Hx Hy Hacc Hblock Hdead Hused Hkind.
    assert (Hsx : mem_node (Node_variable x) (t_has_source T) = true).
    { unfold is_dead_var in Hdead.
      destruct Hkind as [-> | (nk & ->)]; simpl in Hdead;
        apply negb_false_iff in Hdead; exact Hdead. }
    assert (Hux : mem_node (Node_variable x) (t_has_usage T) = true).
    { unfold is_var_used in Hused.
      destruct Hkind as [-> | (nk & ->)]; exact Hused. }
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hhx : has_to_be_unboxed dominated to_unbox (Node_variable x)
                  = false).
    { destruct (has_to_be_unboxed dominated to_unbox (Node_variable x))
        eqn:Hh; [ | reflexivity ].
      unfold has_to_be_unboxed in Hh.
      apply existsb_exists in Hh as ([a x'] & Hin & Htest).
      apply andb_true_iff in Htest as [Hx' Ha].
      apply node_eqb_eq in Hx'. subst x'.
      pose proof (unboxed_fields_of_names_bound F T
                    (bound_code_my_closures e) dominated to_unbox
                    (FlambdaNameAbstractionSpec.fresh_name (names_of e)) a
                    (Node_variable x) Hin Ha)
        as Hb.
      rewrite Hunboxed in Hb. cbn [fst] in Hb. unfold fields_of in Hx.
      exfalso.
      exact (keyed_lookup_none_not_in node_eqb node_eqb_decides _ _ Hx Hb). }
    apply (fields_of_in unboxed) in Hy.
    assert (Hy' : (Node_variable y, fields)
                  ∈ fst (unboxed_fields_of_names F T
                           (bound_code_my_closures e) dominated to_unbox
                           (FlambdaNameAbstractionSpec.fresh_name
                              (names_of e))))
      by (rewrite Hunboxed; exact Hy).
    destruct (unboxed_fields_of_names_dom F T _ dominated to_unbox _ _ _ Hy')
      as (a & next' & Hdom & Ha & Hfields).
    unfold check_unboxed_reads in Hcheck. rewrite forallb_forall in Hcheck.
    specialize (Hcheck _ Hacc). cbn beta iota in Hcheck.
    rewrite Hsx, Hux, Hhx in Hcheck. simpl in Hcheck.
    rewrite forallb_forall in Hcheck. specialize (Hcheck _ Hdom).
    cbn beta iota in Hcheck. rewrite node_eqb_refl, Ha in Hcheck.
    simpl in Hcheck. apply andb_true_iff in Hcheck as [Hnau Hsrc].
    apply negb_true_iff in Hnau.
    assert (Hfs : is_function_slot f = false)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hdata : is_data_field f = true)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hus : mem_pair (Node_variable y) (Node_variable y) (t_usages T)
                  = true)
      by exact (usages_accessor _ _ _ Hcl _ _ _ Hacc Hnau).
    assert (Hhu : mem_node (Node_variable y) (t_has_usage T) = true)
      by exact (has_usage_accessor _ _ _ Hcl _ _ _ Hux Hacc).
    assert (Huses' : mem_node (Node_variable y)
                       (add_usages_through_function_slots F T
                          (bound_code_my_closures e) true
                          (get_direct_usages T [Node_variable y]))
                     = true).
    { rewrite (add_usages_unboxed F T _ unboxed _ _ true Huses Hy).
      apply mem_node_In. apply get_direct_usages_self; assumption. }
    subst fields.
    destruct (get_fields_found F T _ _ _ _ Hacc Huses' Hfs Hux)
      as [Hex Hall].
    apply (mk_unboxed_fields_lookup F T _ dominated to_unbox _ a f Hdata).
    - intros Hno. rewrite Hno in Hsrc. discriminate Hsrc.
    - exact Hex.
    - intros use Hin.
      destruct (Hall use Hin) as [-> | (flow_to & -> & Hw)];
        [ left; reflexivity | right ].
      exists flow_to. split; [ reflexivity | ].
      destruct (forallb _ flow_to) eqn:Hft; [ | reflexivity ].
      rewrite forallb_forall in Hft. rewrite (Hft _ Hw) in Hhx.
      discriminate Hhx. }
  set (dead := filter (fun z =>
                         negb (s_has_usage (solution_of_tables T)
                                 (Node_variable z))
                         && is_none (fields_of unboxed z))
                 (variables_of e)).
  assert (Hdead : ∀ z, among z dead = true ->
            s_has_usage (solution_of_tables T) (Node_variable z) = false).
  { intros z Hz. apply among_In in Hz. unfold dead in Hz.
    apply filter_In in Hz as [_ Hz]. apply andb_true_iff in Hz as [Hz _].
    apply negb_true_iff in Hz. exact Hz. }
  destruct (proj1 (rebuild_uc F (solution_of_tables T)
                     (fun n => mem_node n cannot_change)
                     (unit_code_deps flag u e) pdc
                     (static_names code_ids_bound e) (bound_code_metadata e)
                     unboxed
                     (FlambdaNameAbstractionSpec.fresh_name (names_of e))
                     next dead noted deps cds (unit_kept e) (unit_codes e)
                     (static_names symbols_of_binding e) is_local nr reach
                     ovr rem Hcl Hslots
                     Hfields_read Hreads_found Hdead)
              e empty_env empty_unbox_env [] None (unit_conts u))
    as (Huc & _).
  - split.
    + intros k ps h Hl. discriminate Hl.
    + reflexivity.
    + intros x. reflexivity.
    + reflexivity.
    + reflexivity.
    + reflexivity.
    + intros k. reflexivity.
    + intros k. left. reflexivity.
  - exact Hcalls.
  - split.
    + intros z Hz. split; [ exact (Hfresh z Hz) | ].
      intros Hu Hn. apply among_In. unfold dead. apply filter_In.
      split; [ exact Hz | ]. rewrite Hu, Hn. reflexivity.
    + intros k Hk.
      assert (Hle : (k <= fold_right Pos.max 1 (names_of e))%positive)
        by (apply fold_max_le; unfold names_of; apply in_or_app; right;
            exact Hk).
      unfold FlambdaNameAbstractionSpec.fresh_name in Hnext' |- *.
      split; intros ->; lia.
  - exact Hedges.
  - exact (uc_refines_flow unboxed Hdistinct dead _ _ _ _ _ _ _ Huc Hflow
             Hover Hrem).
Qed.

(* The statement of the fix, for the fragment of
   FlambdaReaperUnboxingRefinesOver2Spec.v, where the flow of the first run
   is as [reach] says and it over-applies only code of [ovr]. *)
Lemma rebuild_unbox_modes_refines_cc_over2
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) (reach : bool -> variable -> code_id -> bool)
    (ovr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_reach_over_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox reach ovr e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  flow_run (fun b f id => reach b f id = true)
    (State
       (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
          (unit_code_deps flag u e) pdc e)
       [] [] empty_static_env empty_heap no_reads) ->
  over_run (fun id => ovr id = true)
    (State
       (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
          (unit_code_deps flag u e) pdc e)
       [] [] empty_static_env empty_heap no_reads) ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode true (fun _ => false)
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox [] pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hcalls Hcheck Hflow Hover.
  apply (rebuild_unbox_modes_refines_cc_over3 flag is_local u e T
           cannot_change dominated to_unbox pdc nr reach ovr (fun _ => true)
           Hslots Hfragment Hclosed
           (calls_unboxing_in_over3_reach nr _ _ _ _ _ _ reach reach ovr
              (fun _ => true) e (fun _ _ _ H => H) Hcalls)
           Hcheck Hflow Hover).
  intros n s _.
  exact (FlambdaReaperPartialGroupsProofs.step_rem_ok_total _ s
           (fun _ => eq_refl)).
Qed.

(* The statement of the fix, for the fragment of
   FlambdaReaperUnboxingRefinesReach2Spec.v, where the flow of the first run
   is as [reach] says. *)
Lemma rebuild_unbox_modes_refines_cc_reach2
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) (reach : bool -> variable -> code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_reach2_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox reach e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  flow_run (fun b f id => reach b f id = true)
    (State
       (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
          (unit_code_deps flag u e) pdc e)
       [] [] empty_static_env empty_heap no_reads) ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode true (fun _ => false)
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox [] pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hcalls Hcheck Hflow.
  apply (rebuild_unbox_modes_refines_cc_over2 flag is_local u e T
           cannot_change dominated to_unbox pdc nr reach (fun _ => true)
           Hslots Hfragment Hclosed
           (calls_unboxing_in_over2_reach nr _ _ _ _ _ _ reach
              (fun _ => true) e Hcalls)
           Hcheck Hflow).
  intros n s _.
  exact (FlambdaReaperPartialGroupsProofs.step_over_ok_total _ s
           (fun _ => eq_refl)).
Qed.

(* The statement of the fix, for the fragment of
   FlambdaReaperUnboxingRefinesReachSpec.v, where the flow of the first run
   is as [reach2 reach] says. *)
Lemma rebuild_unbox_modes_refines_cc_reach
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) (reach : variable -> code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_reach_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox reach e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  flow_run (fun b f id => reach2 reach b f id = true)
    (State
       (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
          (unit_code_deps flag u e) pdc e)
       [] [] empty_static_env empty_heap no_reads) ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode true (fun _ => false)
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox [] pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hcalls.
  exact (rebuild_unbox_modes_refines_cc_reach2 flag is_local u e T
           cannot_change dominated to_unbox pdc nr (reach2 reach) Hslots
           Hfragment Hclosed
           (calls_unboxing_in_flow2_reach nr _ _ _ _ _ _ _ _ e
              (fun _ _ => eq_refl) Hcalls)).
Qed.

(* The statement of the fix, for the fragment of
   FlambdaReaperUnboxingRefinesOver1Spec.v. *)
Lemma rebuild_unbox_modes_refines_cc_over1
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_over1_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode true (fun _ => false)
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox [] pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hcalls Hcheck.
  apply (rebuild_unbox_modes_refines_cc_reach flag is_local u e T
           cannot_change dominated to_unbox pdc nr (fun _ _ => true) Hslots
           Hfragment Hclosed
           (calls_unboxing_in_flow_reach nr _ _ _ _ _ _ _ e Hcalls) Hcheck).
  intros n s _ ap f v id slots _ _ _.
  split; intros _; reflexivity.
Qed.

(* The statement of the fix, for the fragment of
   FlambdaReaperUnboxingRefinesOverSpec.v. *)
Lemma rebuild_unbox_modes_refines_cc_over
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_over_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode true (fun _ => false)
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox [] pdc e).
Proof.
  intros Hslots Hfrag Hclosed Hcalls.
  exact (rebuild_unbox_modes_refines_cc_over1 flag is_local u e T
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
           (calls_unboxing_in_over1_nr nr _ _ _ _ _ _ e Hcalls)).
Qed.

(* The statement of the fix. *)
Lemma rebuild_unbox_modes_refines_cc
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode true (fun _ => false)
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox [] pdc e).
Proof.
  intros Hslots Hfrag Hclosed Hcalls.
  exact (rebuild_unbox_modes_refines_cc_over flag is_local u e T
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
           (calls_unboxing_in_over_nr nr _ _ _ _ _ _ e Hcalls)).
Qed.

Theorem rebuild_unbox_fixed_refines_cc_never_returns :
  rebuild_unbox_fixed_refines_cc_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr.
  exact (rebuild_unbox_modes_refines_cc flag is_local u e T
           cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_fixed_refines_cc :
  rebuild_unbox_fixed_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc.
  exact (rebuild_unbox_fixed_refines_cc_never_returns flag is_local u e T
           cannot_change dominated to_unbox pdc (fun _ => false)).
Qed.

Theorem rebuild_unbox_fixed_tail_refines_cc :
  rebuild_unbox_fixed_tail_refines_cc_statement.
Proof. exact rebuild_unbox_fixed_refines_cc. Qed.

Theorem rebuild_unbox_fixed_local_never_returns :
  rebuild_unbox_fixed_local_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr Hslots
    Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_fixed_refines_cc_never_returns flag is_local u e T
              cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
              Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_fixed_refines_observably_local :
  rebuild_unbox_fixed_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc.
  exact (rebuild_unbox_fixed_local_never_returns flag is_local u e T
           cannot_change dominated to_unbox pdc (fun _ => false)).
Qed.

Theorem rebuild_unbox_fixed_observably_never_returns :
  rebuild_unbox_fixed_observably_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr Hslots
    Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hcalls Hcheck.
  exact (rebuild_unbox_fixed_local_never_returns flag is_local u e T
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_fixed_refines_observably :
  rebuild_unbox_fixed_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc.
  exact (rebuild_unbox_fixed_observably_never_returns flag is_local u e T
           cannot_change dominated to_unbox pdc (fun _ => false)).
Qed.

Theorem rebuild_unbox_fixed_tail_refines_observably_local :
  rebuild_unbox_fixed_tail_refines_observably_local_statement.
Proof. exact rebuild_unbox_fixed_refines_observably_local. Qed.

Theorem rebuild_unbox_fixed_tail_refines_observably :
  rebuild_unbox_fixed_tail_refines_observably_statement.
Proof. exact rebuild_unbox_fixed_refines_observably. Qed.

Theorem fixed_call_example : fixed_call_example_statement.
Proof.
  destruct unboxed_call_example
    as (Hslots & Hfrag & Hstatic & Hrk & Hwf & Hone & Hclosed & Hexact & _
        & Hccc & Hwid & _).
  assert (Hcalls : calls_unboxing CallExample.facts CallExample.tables
                     CallExample.cannot_change CallExample.code_deps
                     CallExample.dominated CallExample.to_unbox
                     CallExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hreads : check_unboxed_reads CallExample.facts CallExample.tables
                     CallExample.dominated CallExample.to_unbox
                   = true)
    by (vm_compute; reflexivity).
  assert (Hrebuilt :
    rebuild_unit_unbox_fixed CallExample.facts CallExample.tables
      CallExample.cannot_change CallExample.code_deps CallExample.dominated
      CallExample.to_unbox true CallExample.unit_expr
    = CallExample.rebuilt)
    by (vm_compute; reflexivity).
  refine (conj Hcalls (conj Hreads (conj Hrebuilt _))).
  rewrite <- Hrebuilt.
  apply (rebuild_unbox_fixed_refines_observably Auto Examples.no_local
           Examples.interface CallExample.unit_expr CallExample.tables
           CallExample.cannot_change CallExample.dominated
           CallExample.to_unbox true); try assumption.
  left. intros f. reflexivity.
Qed.

Theorem fixed_wrapper_example : fixed_wrapper_example_statement.
Proof.
  destruct unboxed_wrapper_example
    as (Hslots & Hfrag & Hstatic & Hrk & Hwf & Hone & Hclosed & Hexact & _
        & Hccc & Hwid & _).
  assert (Hcalls : calls_unboxing WrapperExample.facts WrapperExample.tables
                     WrapperExample.cannot_change WrapperExample.code_deps
                     WrapperExample.dominated WrapperExample.to_unbox
                     WrapperExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hreads : check_unboxed_reads WrapperExample.facts
                     WrapperExample.tables WrapperExample.dominated
                     WrapperExample.to_unbox
                   = true)
    by (vm_compute; reflexivity).
  assert (Hrebuilt :
    rebuild_unit_unbox_fixed WrapperExample.facts WrapperExample.tables
      WrapperExample.cannot_change WrapperExample.code_deps
      WrapperExample.dominated WrapperExample.to_unbox true
      WrapperExample.unit_expr
    = WrapperExample.rebuilt)
    by (vm_compute; reflexivity).
  refine (conj Hcalls (conj Hreads (conj Hrebuilt _))).
  rewrite <- Hrebuilt.
  apply (rebuild_unbox_fixed_refines_observably Auto Examples.no_local
           Examples.interface WrapperExample.unit_expr WrapperExample.tables
           WrapperExample.cannot_change WrapperExample.dominated
           WrapperExample.to_unbox true); try assumption.
  left. intros f. reflexivity.
Qed.

Theorem tail_example : tail_example_statement.
Proof.
  assert (Hcalls : calls_unboxing TailExample.facts TailExample.tables
                     TailExample.cannot_change TailExample.code_deps [] []
                     TailExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hreads : check_unboxed_reads TailExample.facts TailExample.tables
                     [] []
                   = true)
    by (vm_compute; reflexivity).
  assert (Hfixed :
    rebuild_unit_unbox_fixed TailExample.facts TailExample.tables
      TailExample.cannot_change TailExample.code_deps [] [] true
      TailExample.unit_expr
    = TailExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment TailExample.unit_expr = true
    ∧ any_kind_arguments TailExample.unit_expr = true
    ∧ results_kinded [] [] TailExample.unit_expr = true
    ∧ well_formed_unit Examples.interface TailExample.unit_expr = true
    ∧ exits_one_value Examples.interface TailExample.unit_expr = true
    ∧ check_closed Examples.no_local TailExample.facts TailExample.tables
      = true
    ∧ check_field_used_exact Examples.no_local TailExample.facts
        TailExample.tables
      = true
    ∧ check_cc_closed TailExample.facts TailExample.tables
        TailExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        TailExample.facts TailExample.tables
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid).
  assert (Hslots : is_local_slots_only Examples.no_local)
    by (intros f Hf; discriminate Hf).
  split; [ exact Hcalls | split; [ exact Hreads | ] ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hfixed | ].
  rewrite <- Hfixed.
  apply (rebuild_unbox_fixed_refines_observably Auto Examples.no_local
           Examples.interface TailExample.unit_expr TailExample.tables
           TailExample.cannot_change [] [] true); try assumption.
  left. intros f. reflexivity.
Qed.

Theorem exn_example : exn_example_statement.
Proof.
  assert (Hcalls : calls_unboxing ExnExample.facts ExnExample.tables
                     ExnExample.cannot_change ExnExample.code_deps [] []
                     ExnExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hreads : check_unboxed_reads ExnExample.facts ExnExample.tables
                     [] []
                   = true)
    by (vm_compute; reflexivity).
  assert (Hfixed :
    rebuild_unit_unbox_fixed ExnExample.facts ExnExample.tables
      ExnExample.cannot_change ExnExample.code_deps [] [] true
      ExnExample.unit_expr
    = ExnExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment ExnExample.unit_expr = true
    ∧ any_kind_arguments ExnExample.unit_expr = true
    ∧ results_kinded [] [] ExnExample.unit_expr = true
    ∧ well_formed_unit Examples.interface ExnExample.unit_expr = true
    ∧ exits_one_value Examples.interface ExnExample.unit_expr = true
    ∧ check_closed Examples.no_local ExnExample.facts ExnExample.tables
      = true
    ∧ check_field_used_exact Examples.no_local ExnExample.facts
        ExnExample.tables
      = true
    ∧ check_cc_closed ExnExample.facts ExnExample.tables
        ExnExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        ExnExample.facts ExnExample.tables
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid).
  split; [ exact Hcalls | split; [ exact Hreads | ] ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hfixed | ].
  rewrite <- Hfixed.
  apply (rebuild_unbox_fixed_refines_observably Auto Examples.no_local
           Examples.interface ExnExample.unit_expr ExnExample.tables
           ExnExample.cannot_change [] [] true); try assumption.
  - intros f Hf. discriminate Hf.
  - left. intros f. reflexivity.
Qed.

Theorem poison_arg_example : poison_arg_example_statement.
Proof.
  assert (Hcalls : calls_unboxing PoisonArgExample.facts
                     PoisonArgExample.tables PoisonArgExample.cannot_change
                     PoisonArgExample.code_deps PoisonArgExample.dominated
                     PoisonArgExample.to_unbox PoisonArgExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hreads : check_unboxed_reads PoisonArgExample.facts
                     PoisonArgExample.tables PoisonArgExample.dominated
                     PoisonArgExample.to_unbox
                   = true)
    by (vm_compute; reflexivity).
  assert (Hfixed :
    rebuild_unit_unbox_fixed PoisonArgExample.facts PoisonArgExample.tables
      PoisonArgExample.cannot_change PoisonArgExample.code_deps
      PoisonArgExample.dominated PoisonArgExample.to_unbox true
      PoisonArgExample.unit_expr
    = PoisonArgExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment PoisonArgExample.unit_expr = true
    ∧ any_kind_arguments PoisonArgExample.unit_expr = true
    ∧ results_kinded [] [] PoisonArgExample.unit_expr = true
    ∧ well_formed_unit Examples.interface PoisonArgExample.unit_expr = true
    ∧ exits_one_value Examples.interface PoisonArgExample.unit_expr = true
    ∧ check_closed Examples.no_local PoisonArgExample.facts
        PoisonArgExample.tables
      = true
    ∧ check_field_used_exact Examples.no_local PoisonArgExample.facts
        PoisonArgExample.tables
      = true
    ∧ check_cc_closed PoisonArgExample.facts PoisonArgExample.tables
        PoisonArgExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        PoisonArgExample.facts PoisonArgExample.tables
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | split; [ exact Hreads | ] ].
  split; [ exact Hfixed | ].
  rewrite <- Hfixed.
  apply (rebuild_unbox_fixed_refines_observably Auto Examples.no_local
           Examples.interface PoisonArgExample.unit_expr
           PoisonArgExample.tables PoisonArgExample.cannot_change
           PoisonArgExample.dominated PoisonArgExample.to_unbox true);
    try assumption.
  - intros f Hf. discriminate Hf.
  - left. intros f. reflexivity.
Qed.

Theorem escaping_argument_calls : escaping_argument_calls_statement.
Proof.
  unfold escaping_argument_calls_statement. cbv zeta.
  assert (Hcalls :
    calls_unboxing FlambdaReaperUnboxingSpec.escaping_argument_facts
      FlambdaReaperUnboxingSpec.escaping_argument_tables
      FlambdaReaperUnboxingSpec.escaping_argument_cannot_change
      FlambdaReaperUnboxingSpec.escaping_argument_code_deps
      FlambdaReaperUnboxingSpec.escaping_argument_dominated
      (FlambdaReaperUnboxingSpec.escaping_argument_to_unbox true)
      FlambdaReaperUnboxingSpec.escaping_argument_unit
    = true)
    by (vm_compute; reflexivity).
  assert (Hreads :
    check_unboxed_reads FlambdaReaperUnboxingSpec.escaping_argument_facts
      FlambdaReaperUnboxingSpec.escaping_argument_tables
      FlambdaReaperUnboxingSpec.escaping_argument_dominated
      (FlambdaReaperUnboxingSpec.escaping_argument_to_unbox true)
    = true)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment FlambdaReaperUnboxingSpec.escaping_argument_unit = true
    ∧ any_kind_arguments FlambdaReaperUnboxingSpec.escaping_argument_unit
      = true
    ∧ results_kinded [] [] FlambdaReaperUnboxingSpec.escaping_argument_unit
      = true
    ∧ well_formed_unit Examples.interface
        FlambdaReaperUnboxingSpec.escaping_argument_unit
      = true
    ∧ exits_one_value Examples.interface
        FlambdaReaperUnboxingSpec.escaping_argument_unit
      = true
    ∧ check_closed Examples.no_local
        FlambdaReaperUnboxingSpec.escaping_argument_facts
        FlambdaReaperUnboxingSpec.escaping_argument_tables
      = true
    ∧ check_field_used_exact Examples.no_local
        FlambdaReaperUnboxingSpec.escaping_argument_facts
        FlambdaReaperUnboxingSpec.escaping_argument_tables
      = true
    ∧ check_cc_closed FlambdaReaperUnboxingSpec.escaping_argument_facts
        FlambdaReaperUnboxingSpec.escaping_argument_tables
        FlambdaReaperUnboxingSpec.escaping_argument_cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        FlambdaReaperUnboxingSpec.escaping_argument_facts
        FlambdaReaperUnboxingSpec.escaping_argument_tables
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid).
  split; [ exact Hcalls | split; [ exact Hreads | ] ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  apply (rebuild_unbox_fixed_refines_observably Auto Examples.no_local
           Examples.interface FlambdaReaperUnboxingSpec.escaping_argument_unit
           FlambdaReaperUnboxingSpec.escaping_argument_tables
           FlambdaReaperUnboxingSpec.escaping_argument_cannot_change
           FlambdaReaperUnboxingSpec.escaping_argument_dominated
           (FlambdaReaperUnboxingSpec.escaping_argument_to_unbox true) true);
    try assumption.
  - intros f Hf. discriminate Hf.
  - left. intros f. reflexivity.
Qed.

Theorem never_returns_example : never_returns_example_statement.
Proof.
  assert (Hcalls : calls_unboxing_nr NeverExample.nr NeverExample.facts
                     NeverExample.tables NeverExample.cannot_change
                     NeverExample.code_deps [] [] NeverExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hreads : check_unboxed_reads NeverExample.facts
                     NeverExample.tables [] []
                   = true)
    by (vm_compute; reflexivity).
  assert (Hfixed :
    rebuild_unit_unbox_fixed NeverExample.facts NeverExample.tables
      NeverExample.cannot_change NeverExample.code_deps [] [] true
      NeverExample.unit_expr
    = NeverExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment NeverExample.unit_expr = true
    ∧ any_kind_arguments NeverExample.unit_expr = true
    ∧ results_kinded [] [] NeverExample.unit_expr = true
    ∧ well_formed_unit Examples.interface NeverExample.unit_expr = true
    ∧ exits_one_value Examples.interface NeverExample.unit_expr = true
    ∧ check_closed Examples.no_local NeverExample.facts NeverExample.tables
      = true
    ∧ check_field_used_exact Examples.no_local NeverExample.facts
        NeverExample.tables
      = true
    ∧ check_cc_closed NeverExample.facts NeverExample.tables
        NeverExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        NeverExample.facts NeverExample.tables
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid).
  assert (Hslots : is_local_slots_only Examples.no_local)
    by (intros f Hf; discriminate Hf).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | split; [ exact Hreads | ] ].
  split; [ exact Hfixed | ].
  rewrite <- Hfixed.
  apply (rebuild_unbox_fixed_observably_never_returns Auto Examples.no_local
           Examples.interface NeverExample.unit_expr NeverExample.tables
           NeverExample.cannot_change [] [] true NeverExample.nr);
    try assumption.
  left. intros f. reflexivity.
Qed.
