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
   unboxing in the mode of the fix or of its variant ([u_rebuild], the
   section variable [tail]: [rebuild_unbox_modes_refines_cc]), by the
   relation of FlambdaReaperUnboxingRefinesCallsSimProofs.v ([uc]), a
   simulation. [rebuild_uc] builds it by induction on the unit, the
   continuations in scope being described by [pdesc]s ([cinv]: the masks
   of the first rebuild and the decisions of the second are those of the
   descriptions, but for the return continuation of code that keeps its
   calling convention, which has no mask in the first, and for the
   exception continuation of code ([blocked]), which only calls name).
   The two rebuilt terms agree but where the second
   unboxes (as in FlambdaReaperUnboxingRefinesProofs.v), where it keeps a
   parameter with no use (a dead variable: [dead] is the variables of
   the unit with no use that are not unboxed), and at calls
   ([call_rebuilt]): the arguments by [arg_ok] ([call_args_rel]), the
   code by [code_param_ok] ([code_rebuilt]), and the result, case by case
   of the decisions of the callee and of the continuation, returned
   directly or through a wrapper of one rebuild or of both
   ([uc_result_call], [uc_result_call_wrap], [uc_result_call_wraps]).
   Code that keeps its calling convention, and its calls, are related
   apart ([code_rebuilt_nc], [call_rebuilt_nc]): both rebuilds give the
   same arguments ([args_rel2_mask]), and its result, kept with or
   without a use, is returned as the callee keeps it. Where the fix and
   its variant differ, a result kept with no use, the proof is by cases
   on [tail].
   What the second rebuild keeps of an unboxed allocation, and the reads
   that it keeps of unboxed names, are as in
   FlambdaReaperUnboxingRefinesProofs.v, the closures among the uses of
   an unboxed name adding nothing to them by [check_unboxed_uses]
   ([add_usages_unboxed]). The second statement composes the first with
   FlambdaReaperCallingConventionsProofs.rebuild_cc_refines_observably,
   and the examples are the statements applied. *)

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
  FlambdaReaperUnboxingRefinesCallsSimProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsProofs FlambdaNameAbstractionSpec.
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

Definition agree_on2 (l : list continuation) (K K' : finite_map kshape)
    : Prop :=
  ∀ k : continuation, k ∈ l -> lookup k K' = lookup k K.

Lemma uc_handlers_keys (K : finite_map kshape)
    (hs hs' : list (continuation * continuation_handler))
    (S : finite_map kshape) :
  uc_handlers M dead CS K hs hs' S -> map fst S = map fst hs.
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

Lemma exn_ok_agree (K K' : finite_map kshape) (e : continuation) :
  lookup e K' = lookup e K -> exn_ok K e -> exn_ok K' e.
Proof. intros H [He | He]; [ left | right ]; rewrite H; exact He. Qed.

(* The shapes of the continuations not free in a term do not matter. *)
Lemma uc_agree :
  ∀ (K : finite_map kshape) (p q : expr),
    uc M dead CS K p q ->
    ∀ K' : finite_map kshape,
      agree_on2 (free_continuations_of_expr p) K K' -> uc M dead CS K' p q.
Proof.
  apply (uc_mut M dead CS
           (fun K p q _ =>
              ∀ K', agree_on2 (free_continuations_of_expr p) K K' ->
                    uc M dead CS K' p q)
           (fun K hs hs' S _ =>
              ∀ K',
                agree_on2
                  (flat_map (fun '(_, h) => free_continuations_of_handler h)
                     hs)
                  K K' ->
                uc_handlers M dead CS K' hs hs' S)
           (fun _ _ _ => True)).
  - intros K x kind d body body' Hx Hd Hnh _ IH K' Hagree.
    apply uc_let; [ exact Hx | exact Hd | exact Hnh | apply IH, Hagree ].
  - intros K x d fields l body body' Hf Hxf Hl Hoff _ IH K' Hagree.
    apply uc_alloc with (fields := fields); try assumption.
    apply IH, Hagree.
  - intros K x y fields fields_y ls body body' Hf Hxf Hy Hne Hl _ IH K'
      Hagree.
    apply (uc_alias M dead CS K' x y fields fields_y ls); try assumption.
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
  - intros K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' _ IHh _ IHb K'
      Hagree.
    apply (uc_let_cont M dead CS K' k sh ps ps' md md' h h' body body');
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
    apply (uc_let_cont_rec M dead CS K' S);
      [ apply IHhs | exact Hst | apply IHb ];
      apply (agree_app2 S K K' (map fst hs)); try exact Hkeys;
      intros k Hk; apply Hagree; simpl; apply in_without_iff;
      apply in_without_iff in Hk as [Hk Hnot]; (split; [ | exact Hnot ]);
      apply in_or_app; [ left | right ]; exact Hk.
  - intros K ac ac' (Hk & Htrap & Hargs) K' Hagree.
    apply uc_apply_cont. split; [ exact Hk | split; [ exact Htrap | ] ].
    unfold kshape_at.
    rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)).
    exact Hargs.
  - intros K scrutinee arms arms' Hs Harms K' Hagree.
    apply uc_switch; [ exact Hs | ]. simpl in Hagree. clear Hs.
    induction Harms as [ | [d ac] [d' ac'] arms arms' [Hd (Hk & Htrap & Hargs)]
                         _ IH];
      constructor.
    + split; [ exact Hd | ]. split; [ exact Hk | split; [ exact Htrap | ] ].
      simpl in Hk, Htrap, Hargs |- *. unfold kshape_at in Hargs |- *.
      rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)). exact Hargs.
    + apply IH. intros k Hin. apply Hagree. simpl. right. apply in_or_app.
      right. exact Hin.
  - intros K K' _. apply uc_invalid.
  - intros K ap ap' cs k sh Hcall Hexn Hk Hk' Hlk Ht K' Hagree.
    apply (uc_call M dead CS K' ap ap' cs k sh); try assumption.
    + apply (exn_ok_agree K); [ | exact Hexn ].
      apply Hagree. simpl. right. left. reflexivity.
    + rewrite <- Hlk. apply Hagree. left. exact Hk.
  - intros K ap ap' cs k sh w wps wmd wargs Hcall Hexn Hk Hw Hwe Hlk Hok K'
      Hagree.
    apply (uc_call_wrap M dead CS K' ap ap' cs k sh w wps wmd wargs);
      try assumption.
    + apply (exn_ok_agree K); [ | exact Hexn ].
      apply Hagree. simpl. right. left. reflexivity.
    + rewrite <- Hlk. apply Hagree. left. exact Hk.
  - intros K ap ap' cs k sh KS wc cps cmd cargs w wps wmd wargs Hcall Hexn
      Hwc Hw Hwce Hwe Hlk Hcmd Hok K' Hagree.
    apply (uc_call_wraps M dead CS K' ap ap' cs k sh KS wc cps cmd cargs w wps
             wmd wargs); try assumption.
    + apply (exn_ok_agree K); [ | exact Hexn ].
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
  - intros K K' _. apply uc_handlers_nil.
  - intros K k sh ps ps' md md' h h' hs hs' S Hps Hmd Hmd' _ IHh _ IHhs K'
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

(* The fix, or its variant. *)
Variable tail : bool.
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
  rebuild_expr_u true tail F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu [] en e.

(* The shapes of the code of the unit. *)
Definition CSf (id : code_id) : option code_shape :=
  match lookup id codes with
  | Some md =>
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
           (arity_of_decisions_u rd))
    | Not_changing_u, Some (pds, rds) =>
      Some
        (Code_shape (flat_map pshape_of pds) (flat_map pshape_of rds)
           (flat_map kept_kind rds) (cm_params_arity md)
           (cm_result_arity md))
    | _, None => None
    end
  | None => None
  end.

Record cinv (C : finite_map (list pdesc * bool)) (blocked : list continuation)
    (en : rebuild_env) (enu : unbox_env) : Prop := {
  ci_ok : ∀ (k : continuation) (ps : list pdesc) (h : bool),
    lookup k C = Some (ps, h) -> forallb pdesc_ok ps = true;
  ci_kinds : env_kinds (ue_env enu) = env_kinds en;
  ci_kinds_ok : ∀ x : variable, kind_ok (kind_of_variable en x) = true;
  ci_proj : env_projections en = [];
  ci_proj_u : env_projections (ue_env enu) = [];
  ci_drop : ue_drop_projections enu = false;
  ci_dec : ∀ k : continuation,
    among k blocked = false ->
    lookup k (ue_decisions enu)
    = option_map (fun e => map pd_decision (fst e)) (lookup k C);
  (* The return continuation of code that keeps its calling convention has
     no mask in the first run, which keeps all its parameters. *)
  ci_mask : ∀ k : continuation,
    among k blocked = false ->
    lookup k (env_conts en)
    = option_map (fun e => map pd_keep (fst e)) (lookup k C)
    ∨ lookup k (env_conts en) = None
      ∧ ∃ (ps : list pdesc) (h : bool),
          lookup k C = Some (ps, h) ∧ forallb pd_keep ps = true;
  ci_blocked : ∀ k : continuation,
    among k blocked = true -> ∃ ps : list pdesc, lookup k C = Some (ps, false)
}.

Definition uc_result (K : finite_map kshape) (fc0 : list continuation)
    (p q : expr) : Prop :=
  uc unboxed dead CSf K p q
  ∧ free_continuations_of_expr p = free_continuations_of_expr q
  ∧ binds_statics p = binds_statics q
  ∧ incl (free_continuations_of_expr p) fc0.

Definition var_ok (z : variable) : Prop :=
  fresh z
  ∧ (s_has_usage sol (Node_variable z) = false ->
     fields_of unboxed z = None -> among z dead = true).

Lemma uc_result_invalid (K : finite_map kshape) (fc0 : list continuation) :
  uc_result K fc0 Invalid Invalid.
Proof.
  split; [ constructor | split; [ reflexivity | split; [ reflexivity | ] ] ].
  intros k [].
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

Lemma cinv_bind (C : finite_map (list pdesc * bool))
    (blocked : list continuation) (en : rebuild_env) (enu : unbox_env)
    (bound : list (variable * flambda_kind)) :
  cinv C blocked en enu ->
  (∀ y kind, (y, kind) ∈ bound -> kind_ok kind = true) ->
  cinv C blocked (bind_variables en bound) (ubind_variables enu bound).
Proof.
  intros [Hok Hk Hkok Hp Hpu Hd Hdec Hmask Hbl] Hb. split.
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
  - exact Hbl.
Qed.

Lemma lookup_map_ps {B : Type} (g : list pdesc -> B)
    (D : finite_map (list pdesc * bool)) (k : continuation) :
  lookup k (map (fun '(k, (ps, _)) => (k, g ps)) D)
  = option_map (fun e => g (fst e)) (lookup k D).
Proof.
  induction D as [ | [k' [ps h]] D IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma cinv_conts (C : finite_map (list pdesc * bool))
    (blocked : list continuation) (en : rebuild_env) (enu : unbox_env)
    (D : finite_map (list pdesc * bool)) :
  cinv C blocked en enu ->
  (∀ k ps h, (k, (ps, h)) ∈ D -> forallb pdesc_ok ps = true) ->
  cinv (D ++ C) (without (map fst D) blocked)
    (bind_continuations en (map (fun '(k, (ps, _)) => (k, map pd_keep ps)) D))
    (ubind_continuations enu
       (map (fun '(k, (ps, _)) => (k, map pd_decision ps)) D)).
Proof.
  intros [Hok Hk Hkok Hp Hpu Hd Hdec Hmask Hbl] HD. split.
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
  - intros k Hb. cbn [ue_decisions ubind_continuations].
    rewrite !lookup_app.
    rewrite (lookup_map_ps (map pd_decision) D k).
    destruct (lookup k D) as [[ps h] | ] eqn:HlD; [ reflexivity | ].
    apply Hdec. destruct (among k blocked) eqn:Hkb; [ | reflexivity ].
    exfalso. assert (Hin : k ∈ without (map fst D) blocked).
    { apply in_without_iff. split; [ apply among_In, Hkb | ].
      intros HinD. exact (lookup_keys k D HinD HlD). }
    apply among_In in Hin. rewrite Hin in Hb. discriminate.
  - intros k Hb. cbn [env_conts bind_continuations].
    rewrite !lookup_app.
    rewrite (lookup_map_ps (map pd_keep) D k).
    destruct (lookup k D) as [[ps h] | ] eqn:HlD; [ left; reflexivity | ].
    apply Hmask. destruct (among k blocked) eqn:Hkb; [ | reflexivity ].
    exfalso. assert (Hin : k ∈ without (map fst D) blocked).
    { apply in_without_iff. split; [ apply among_In, Hkb | ].
      intros HinD. exact (lookup_keys k D HinD HlD). }
    apply among_In in Hin. rewrite Hin in Hb. discriminate.
  - intros k Hb. apply among_In, in_without_iff in Hb as [Hb Hne].
    apply among_In in Hb. destruct (Hbl k Hb) as (ps & Hl).
    exists ps. rewrite lookup_app.
    destruct (lookup k D) as [p | ] eqn:HlD; [ | exact Hl ].
    exfalso. apply Hne. apply (keyed_lookup_in Pos.eqb Pos.eqb_eq) in HlD.
    exact (in_map fst D (k, p) HlD).
Qed.

(* A continuation applied is not blocked. *)
Lemma not_blocked_ac (C : finite_map (list pdesc * bool))
    (blocked : list continuation) (en : rebuild_env) (enu : unbox_env)
    (ac : apply_cont) :
  cinv C blocked en enu ->
  apply_cont_ok2 sol unboxed (env_kinds en) C ac = true ->
  among (ac_continuation ac) blocked = false.
Proof.
  intros Hci Hok. destruct (among _ blocked) eqn:E; [ | reflexivity ].
  destruct (ci_blocked _ _ _ _ Hci _ E) as (ps & Hl).
  unfold apply_cont_ok2 in Hok. rewrite Hl in Hok. discriminate Hok.
Qed.

(* The exception continuation of a call is a way out, or that of the code
   the call is in. *)
Lemma exn_cont_ok_K (C : finite_map (list pdesc * bool)) (e : continuation) :
  exn_cont_ok C e = true -> exn_ok (K_of C) e.
Proof.
  unfold exn_cont_ok, exn_ok. rewrite lookup_K_of. intros H.
  destruct (lookup e C) as [[ps hb] | ]; [ right | left; reflexivity ].
  cbn iota beta in H.
  repeat match type of H with
  | match ?x with _ => _ end = true => destruct x; try discriminate H
  end.
  reflexivity.
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
        exists (Const (Poison kind') :: rest).
        split; [ reflexivity | ]. cbn [filter_kept app].
        apply args_rel2_dead; [ left; reflexivity | exact Hrel ].
Qed.

Lemma rewrite_apply_cont_rel2 (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (blocked : list continuation)
    (ac : apply_cont) :
  cinv C blocked en enu -> among (ac_continuation ac) blocked = false ->
  (∀ z, z ∈ variables_in_apply_cont ac -> fresh z) ->
  apply_cont_ok2 sol unboxed (env_kinds en) C ac = true ->
  (rewrite_apply_cont sol en ac = None
   ∧ rewrite_apply_cont_u sol unboxed enu ac = None)
  ∨ ∃ ac_p ac_u : apply_cont,
      rewrite_apply_cont sol en ac = Some ac_p
      ∧ rewrite_apply_cont_u sol unboxed enu ac = Some ac_u
      ∧ ac_rel2 unboxed dead (K_of C) ac_p ac_u.
Proof.
  intros Hci Hb Hfresh Hok. unfold rewrite_apply_cont, rewrite_apply_cont_u.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ left; split; reflexivity | right ].
  unfold apply_cont_ok2 in Hok.
  assert (Hmap : map (rewrite_simple sol (ue_env enu)) (ac_args ac)
                 = map (rewrite_simple sol en) (ac_args ac))
    by (apply map_ext; intros s; apply rewrite_simple_kinds, (ci_kinds _ _ _ _
                                                               Hci)).
  pose proof (ci_mask _ _ _ _ Hci _ Hb) as Hm.
  rewrite (ci_dec _ _ _ _ Hci _ Hb).
  unfold ac_rel2, kshape_at.
  destruct (lookup (ac_continuation ac) C) as [[ps [ | ]] | ] eqn:HCk;
    [ | discriminate Hok | ].
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
    + intros x. unfold kind_of_variable. rewrite (ci_kinds _ _ _ _ Hci).
      exact (ci_kinds_ok _ _ _ _ Hci x).
    + exact Hfresh.
    + exact (ci_ok _ _ _ _ Hci _ _ _ HCk).
    + rewrite (ci_kinds _ _ _ _ Hci). exact Hok.
    + rewrite Hget. eexists _, _.
      split; [ reflexivity | split; [ reflexivity | ] ].
      cbn [ac_continuation ac_args ac_trap_action].
      split; [ reflexivity | split; [ reflexivity | ] ].
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
    rewrite lookup_K_of, HCk. cbn [option_map].
    apply args_rel2_beyond, rewrite_plains2;
      [ exact (ci_kinds_ok _ _ _ _ Hci) | exact Hstrong ].
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
         (rebuild_binding_u true tail F sol cannot_change code_deps pdc
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
        (rebuild_handler_u true tail F sol cannot_change code_deps pdc
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
             rebuild_handler_u true tail F sol cannot_change code_deps pdc
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
  rebuild_handler_u true tail F sol cannot_change code_deps pdc unit_code_ids
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
    (C : finite_map (list pdesc * bool)) (blocked : list continuation)
    (ctx : option code_id) (ce : finite_map (list node)),
    cinv C blocked en enu ->
    calls_expr F sol cannot_change code_deps unit_code_ids codes unboxed
      (env_kinds en) C e
    = true ->
    names_ok e -> edges ctx ce e ->
    uc_result (K_of C) (free_continuations_of_expr e) (cc_rebuild en e)
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
   one group. *)
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
  ap_call_kind ap = Direct id ->
  code_id_actually_called F sol ap = Some id ->
  rewrite_call_kind F sol pdc unit_code_ids ap = Direct id.
Proof.
  intros Hk Hc. unfold code_id_actually_called in Hc.
  unfold rewrite_call_kind. rewrite Hk in Hc |- *.
  destruct (ap_callee ap); try discriminate Hc;
    destruct (code_id_actually_directly_called F sol _) as [ids | ];
    try discriminate Hc; rewrite Hc; reflexivity.
Qed.

Lemma simple_is_dead_kinds (en1 en2 : rebuild_env) (s : simple) :
  env_kinds en1 = env_kinds en2 ->
  simple_is_dead sol en1 s = simple_is_dead sol en2 s.
Proof.
  intros H. destruct s; simpl; [ | reflexivity | reflexivity ].
  unfold kind_of_variable. rewrite H. reflexivity.
Qed.

Lemma group_by_one {A : Type} (c : arity_component) (l : list A) :
  length l <= length (unarize_component c) ->
  group_by_parameter [c] l = [l].
Proof.
  intros H. unfold group_by_parameter, unarize_per_parameter. simpl.
  rewrite firstn_all2 by exact H. reflexivity.
Qed.

Lemma unarize_one (c : arity_component) : unarize [c] = unarize_component c.
Proof. unfold unarize. simpl. apply app_nil_r. Qed.

Lemma unarize_singletons (ks : list flambda_kind) :
  unarize [Unboxed_product (map Singleton ks)] = ks.
Proof.
  unfold unarize. simpl. rewrite app_nil_r.
  induction ks as [ | k ks IH]; simpl; [ reflexivity | ]. rewrite IH.
  reflexivity.
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

Lemma bind_unbox_wrapper (fr fr' : unboxed_fields) (v : value)
    (ws : list value) (wv : env) :
  fields_paths fr' = fields_paths fr -> NoDup (leaf_variables fr') ->
  fields_vals fr v ws ->
  ∃ hv : env,
    bind_params (leaf_variables fr') ws wv = Some hv ∧ fields_hold hv fr' v.
Proof.
  intros Hp Hnd Hws.
  assert (Hws' : fields_vals fr' v ws) by (unfold fields_vals; rewrite Hp;
                                            exact Hws).
  assert (Hlen : length (leaf_variables fr') = length ws).
  { unfold leaf_variables, fields_vals in *. rewrite length_map.
    rewrite <- (Forall2_length Hws'). unfold fields_paths, fields_leaves.
    rewrite ufield_paths_length. reflexivity. }
  destruct (bind_params_some _ ws wv Hlen) as (hv & Hb).
  exists hv. split; [ exact Hb | ].
  exact (bind_fields_hold_gen fr' v ws wv hv Hnd Hws' Hb).
Qed.

Lemma unbox_kinds (fr fr' : unboxed_fields) (v : value) (ws : list value) :
  fields_paths fr' = fields_paths fr -> fields_vals fr v ws ->
  map kind_of_value ws = map snd (fields_leaves fr').
Proof.
  intros Hp Hws. rewrite (fields_vals_kinds fr v ws Hws), leaves_kinds, Hp.
  reflexivity.
Qed.

Lemma kinds_eqb_of (l l' : list flambda_kind) :
  l = l' -> kinds_eqb l l' = true.
Proof. intros ->. apply kinds_eqb_refl. Qed.

(* A variable of a result that the callee keeps, bound by the wrapper. *)
Lemma bind_one (x : variable) (v : value) (wv : env) :
  bind_params [x] [v] wv = Some ((x, v) :: wv).
Proof. reflexivity. Qed.



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
    | Not_changing_calling_convention => Code params mc ret exn md body
    | Changing_calling_convention params_decisions return_decisions =>
      Code (filter_kept (map is_kept params_decisions) params) mc ret exn
        (changed_code_metadata md params_decisions return_decisions) body
    end.
Proof. reflexivity. Qed.

Lemma u_rebuild_code (id : code_id) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (body : expr) :
  rebuild_code_u true tail F sol cannot_change code_deps pdc unit_code_ids codes
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
    | Not_changing_u => Code params mc ret exn md body
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
Lemma code_rebuilt_nc (id : code_id) (c : code) :
  rebuild_ok_expr (code_body c) ->
  calls_code F sol cannot_change code_deps unit_code_ids codes unboxed id c
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
    rebuild_code_u true tail F sol cannot_change code_deps pdc unit_code_ids
      codes unboxed ffu [] id c
  in
  (∃ cs : code_shape,
     code_frame unboxed dead CSf id ccc cu cs
     ∧ uc unboxed dead CSf (code_K ccc cs) (code_body ccc) (code_body cu))
  ∧ free_continuations_of_code ccc = free_continuations_of_code cu
  ∧ incl (free_continuations_of_code ccc) (free_continuations_of_code c).
Proof.
  destruct c as [params mc ret exn md body].
  intros IH Hok Hvars Hconts Hedges Hnc. cbv zeta.
  cbn [metadata_of_code] in Hnc.
  cbn [calls_code] in Hok.
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
  assert (Hci : cinv [(ret, (rds, true)); (exn, (exn_pdescs, false))] [exn]
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
      injection Hl as <- <-. reflexivity.
    - reflexivity.
    - intros x. apply kind_of_bind; [ intros y; reflexivity | ].
      exact Hbound_ok.
    - reflexivity.
    - reflexivity.
    - reflexivity.
    - intros k Hk. cbn [ue_decisions ubind_variables ubind_variables_then
                         ubind_set ubind_continuations empty_unbox_env app].
      rewrite !lookup_cons. destruct (Pos.eqb k ret) eqn:Hkr;
        [ cbn; rewrite Hrdec; reflexivity | ].
      destruct (Pos.eqb k exn) eqn:Hke; [ | reflexivity ].
      apply Pos.eqb_eq in Hke. subst k. rewrite among_cons, Pos.eqb_refl in Hk.
      discriminate Hk.
    - intros k Hk. cbn [env_conts bind_variables bind_variables_then
                         empty_env].
      rewrite !lookup_cons. destruct (Pos.eqb k ret) eqn:Hkr.
      + right. split; [ reflexivity | ]. exists rds, true.
        split; [ reflexivity | ]. rewrite <- Hrds. apply all_kept_pdescs.
      + destruct (Pos.eqb k exn) eqn:Hke; [ | left; reflexivity ].
        apply Pos.eqb_eq in Hke. subst k.
        rewrite among_cons, Pos.eqb_refl in Hk. discriminate Hk.
    - intros k Hk. rewrite among_cons, orb_false_r in Hk.
      apply Pos.eqb_eq in Hk. subst k. exists exn_pdescs.
      rewrite !lookup_cons, Pos.eqb_refl.
      rewrite (proj2 (Pos.eqb_neq exn ret) (fun H => Hre (eq_sym H))).
      reflexivity. }
  assert (Hcs : CSf id
                = Some
                    (Code_shape (flat_map pshape_of pds)
                       (flat_map pshape_of rds) (flat_map kept_kind rds)
                       (cm_params_arity md) (cm_result_arity md))).
  { unfold CSf. rewrite Hcodes, Hnc, Hpd'. reflexivity. }
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
            binds_statics body' = binds_statics body ->
            code_frame unboxed dead CSf id
              (Code params mc ret exn md body)
              (Code params mc ret exn md body')
              (Code_shape
                 (flat_map pshape_of
                    (pdescs_of (map (fun k => Decided (Keep k)) kinds_p)
                       (map is_kept (map Keep kinds_p)) kinds_p))
                 (flat_map pshape_of rds) (flat_map kept_kind rds)
                 (cm_params_arity md) (cm_result_arity md))).
  { intros b b' Hbs. split; [ exact Hcs | ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | ].
    split; [ exact Htup | split; [ exact Htup | ] ].
    split; [ reflexivity | split; [ exact Hmcf | ] ].
    split; [ exact (proj1 (Hvars mc (or_introl eq_refl))) | ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ exact Hbs | ].
    exists sh, pk, pk'. split; [ exact Hpk | ].
    split; [ exact Hpk' | split; [ exact Hrel | exact Hfits ] ]. }
  destruct (existsb (fun '(x, kind) => is_dead_var sol x kind) bound)
    eqn:Hdead.
  - split; [ | split; reflexivity || (intros k []) ].
    eexists. split; [ apply Hframe; reflexivity | constructor ].
  - destruct (IH _ _ [(ret, (rds, true)); (exn, (exn_pdescs, false))] [exn]
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
    + eexists. split; [ apply Hframe; symmetry; exact IHbs | ].
      cbn [code_body]. exact IHuc.
    + cbn [free_continuations_of_code]. f_equal. exact IHfc.
    + intros k Hk. cbn [free_continuations_of_code] in Hk |- *.
      apply in_without_iff in Hk as [Hk Hne].
      apply in_without_iff. split; [ exact (IHincl k Hk) | exact Hne ].
Qed.

(* Code of the unit, as the two rebuilds make it. *)
Lemma code_rebuilt (id : code_id) (c : code) :
  rebuild_ok_expr (code_body c) ->
  calls_code F sol cannot_change code_deps unit_code_ids codes unboxed id c
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
    rebuild_code_u true tail F sol cannot_change code_deps pdc unit_code_ids
      codes unboxed ffu [] id c
  in
  (∃ cs : code_shape,
     code_frame unboxed dead CSf id ccc cu cs
     ∧ uc unboxed dead CSf (code_K ccc cs) (code_body ccc) (code_body cu))
  ∧ free_continuations_of_code ccc = free_continuations_of_code cu
  ∧ incl (free_continuations_of_code ccc) (free_continuations_of_code c).
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
  cbn [calls_code] in Hok.
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
  assert (Hci : cinv [(ret, (rds, true)); (exn, (exn_pdescs, false))] [exn]
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
      injection Hl as <- <-. reflexivity.
    - reflexivity.
    - intros x. apply kind_of_bind; [ intros y; reflexivity | ].
      exact Hbound_ok.
    - reflexivity.
    - reflexivity.
    - reflexivity.
    - intros k Hk. cbn [ue_decisions ubind_variables ubind_variables_then
                         ubind_set ubind_continuations empty_unbox_env app].
      rewrite !lookup_cons. destruct (Pos.eqb k ret) eqn:Hkr;
        [ cbn; rewrite Hrdec; reflexivity | ].
      destruct (Pos.eqb k exn) eqn:Hke; [ | reflexivity ].
      apply Pos.eqb_eq in Hke. subst k. rewrite among_cons, Pos.eqb_refl in Hk.
      discriminate Hk.
    - intros k Hk. left. cbn [env_conts bind_variables bind_variables_then
                               bind_continuations empty_env app].
      rewrite !lookup_cons. destruct (Pos.eqb k ret);
        [ cbn; rewrite Hrkeep; reflexivity | ].
      destruct (Pos.eqb k exn) eqn:Hke; [ | reflexivity ].
      apply Pos.eqb_eq in Hke. subst k. rewrite among_cons, Pos.eqb_refl in Hk.
      discriminate Hk.
    - intros k Hk. rewrite among_cons, orb_false_r in Hk.
      apply Pos.eqb_eq in Hk. subst k. exists exn_pdescs.
      rewrite !lookup_cons, Pos.eqb_refl.
      rewrite (proj2 (Pos.eqb_neq exn ret) (fun H => Hre (eq_sym H))).
      reflexivity. }
  assert (Hcs : CSf id
                = Some
                    (Code_shape (flat_map pshape_of pds)
                       (flat_map pshape_of rds)
                       (flat_map kept_kind rds)
                       (arity_of_grouped_decisions_u
                          (prepend_first [Decided Delete]
                             (group_by_parameter (cm_params_arity md) pd)))
                       (arity_of_decisions_u rd))).
  { unfold CSf. rewrite Hcodes, Hchu'. unfold code_pdescs.
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
    split; [ | split; reflexivity || (intros k []) ].
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
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | split; [ exact Hmcf | ] ].
        split; [ exact (proj1 (Hvars mc (or_introl eq_refl))) | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | ].
        exists sh, pk, pk'. split; [ exact Hpk | ].
        split; [ cbn [code_params decision_params app];
                 rewrite flat_map_decision_params; exact Hpk' | ].
        split; [ exact Hrel | exact Hfits ].
      * constructor.
  - destruct (IH _ _ [(ret, (rds, true)); (exn, (exn_pdescs, false))] [exn]
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
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ reflexivity | split; [ exact Hmcf | ] ].
        split; [ exact (proj1 (Hvars mc (or_introl eq_refl))) | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ cbn [code_body]; symmetry; exact IHbs | ].
        exists sh, pk, pk'. split; [ exact Hpk | ].
        split; [ cbn [code_params decision_params app];
                 rewrite flat_map_decision_params; exact Hpk' | ].
        split; [ exact Hrel | exact Hfits ].
      * cbn [code_body]. exact IHuc.
    + cbn [free_continuations_of_code]. f_equal. exact IHfc.
    + intros k Hk. cbn [free_continuations_of_code] in Hk |- *.
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
  rebuild_binding_u true tail F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu [] enu (Bind_code id c)
  = if is_code_id_used sol id
    then [Bind_code id (rebuild_code_u true tail F sol cannot_change code_deps
                          pdc unit_code_ids codes unboxed ffu [] id c)]
    else [Bind_deleted_code id].
Proof. reflexivity. Qed.

Lemma cc_rebuild_binding_deleted (en : rebuild_env) (id : code_id) :
  rebuild_binding_cc F sol cannot_change code_deps pdc unit_code_ids codes ff
    en (Bind_deleted_code id)
  = [Bind_deleted_code id].
Proof. reflexivity. Qed.

Lemma u_rebuild_binding_deleted (enu : unbox_env) (id : code_id) :
  rebuild_binding_u true tail F sol cannot_change code_deps pdc unit_code_ids
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
  rebuild_binding_u true tail F sol cannot_change code_deps pdc unit_code_ids
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
    (C : finite_map (list pdesc * bool)) (blocked : list continuation)
    (bindings : list static_binding) :
  (∀ (id : code_id) (c : code),
     Bind_code id c ∈ bindings -> rebuild_ok_expr (code_body c)) ->
  cinv C blocked en enu ->
  forallb
    (calls_binding F sol cannot_change code_deps unit_code_ids codes unboxed)
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
      (rebuild_binding_u true tail F sol cannot_change code_deps pdc
         unit_code_ids codes unboxed ffu [] enu)
      bindings
  in
  uc_bindings unboxed dead CSf bcc bu
  ∧ flat_map free_continuations_of_binding bcc
    = flat_map free_continuations_of_binding bu
  ∧ existsb binding_binds bcc = existsb binding_binds bu
  ∧ incl (flat_map free_continuations_of_binding bcc)
      (flat_map free_continuations_of_binding bindings).
Proof.
  intros IHcode Hci Hok Hvars Hconts Hedges. cbv zeta.
  induction bindings as [ | b bs IHbs].
  - split; [ constructor | ].
    split; [ reflexivity | split; [ reflexivity | intros k [] ] ].
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
    + cbn [calls_binding] in Hb.
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
        split; [ apply (uc_bindings_code _ _ _ _ _ _ cs); assumption | ].
        cbn [flat_map free_continuations_of_binding existsb binding_binds
             orb app].
        split; [ f_equal; [ exact Hcfc | exact Hfc ] | ].
        split; [ reflexivity | ].
        intros k Hk. apply in_app_or in Hk as [Hk | Hk]; apply in_or_app;
          [ left; exact (Hcincl k Hk) | right; exact (Hincl k Hk) ].
      * cbn [app].
        split; [ apply uc_bindings_deleted; exact Hub | ].
        cbn [flat_map free_continuations_of_binding existsb binding_binds
             orb app].
        split; [ exact Hfc | split; [ exact Hbb | ] ].
        intros k Hk. apply in_or_app. right. exact (Hincl k Hk).
    + rewrite cc_rebuild_binding_deleted, u_rebuild_binding_deleted.
      cbn [app].
      split; [ apply uc_bindings_deleted; exact Hub | ].
      cbn [flat_map free_continuations_of_binding existsb binding_binds orb
           app].
      split; [ exact Hfc | split; [ exact Hbb | ] ].
      intros k Hk. exact (Hincl k Hk).
    + cbn [calls_binding] in Hb.
      destruct const as [ | | tag fields | | | ]; cbn [local_binding] in Hb;
        try discriminate Hb.
      assert (Hfields : forallb (plain_simple unboxed) fields = true).
      { apply (strong_simples2 fields Hb). intros z Hz.
        exact (proj1 (Hvars z (in_or_app _ _ _ (or_introl Hz)))). }
      rewrite cc_rebuild_binding_block, u_rebuild_binding_block.
      destruct (is_symbol_used sol sym).
      * cbn [app].
        rewrite (rebuild_fields_kinds sol_u (ue_env enu) en _ fields
                   (ci_kinds _ _ _ _ Hci) 0),
          (rebuild_fields_u sol unboxed en _ fields Hfields 0).
        split; [ apply uc_bindings_symbol; [ | exact Hub ] | ].
        { cbn [local_binding2]. apply rebuild_fields_plain2;
            [ exact (ci_kinds_ok _ _ _ _ Hci) | exact Hfields ]. }
        cbn [flat_map free_continuations_of_binding existsb binding_binds
             orb app].
        split; [ exact Hfc | split; [ reflexivity | ] ].
        intros k Hk. exact (Hincl k Hk).
      * cbn [app]. split; [ exact Hub | ].
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
  call_rel unboxed dead CSf ap ap' cs ->
  call_rel unboxed dead CSf (with_continuation ap w) (with_continuation ap' w')
    cs.
Proof. intros H. exact H. Qed.

Lemma uc_result_call (K : finite_map kshape) (ap0 ap ap' : apply_expr)
    (cs : code_shape) (k : continuation) (sh : list pshape) :
  call_rel unboxed dead CSf ap ap' cs -> exn_ok K (ap_exn_continuation ap) ->
  ap_continuation ap = k -> ap_continuation ap' = k ->
  ap_exn_continuation ap = ap_exn_continuation ap0 ->
  ap_continuation ap0 = k ->
  lookup k K = Some (Kshape sh (cs_result_kinds cs) true) ->
  transport (cs_results cs) sh (cs_result_kinds cs) ->
  uc_result K (free_continuations_of_expr (Apply ap0)) (Apply ap) (Apply ap').
Proof.
  intros Hcall Hexn Hk Hk' He Hk0 Hlk Ht.
  split; [ exact (uc_call unboxed dead CSf K ap ap' cs k sh Hcall Hexn Hk Hk'
                    Hlk Ht) | ].
  pose proof Hcall as (id & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & He' &
                       _).
  cbn [free_continuations_of_expr binds_statics].
  rewrite Hk, Hk', He'. split; [ reflexivity | split; [ reflexivity | ] ].
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
  call_rel unboxed dead CSf ap ap' cs -> exn_ok K (ap_exn_continuation ap) ->
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
  split; [ exact (uc_call_wrap unboxed dead CSf K ap ap' cs k sh w wps wmd
                    wargs Hcall Hexn Hk Hw Hwe Hlk Hok) | ].
  pose proof Hcall as (id & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & He' &
                       _).
  rewrite fc_wrap by (try exact Hw; rewrite He'; intros H; apply Hwe;
                      symmetry; exact H).
  cbn [free_continuations_of_expr binds_statics].
  rewrite Hk, He'. split; [ reflexivity | split; [ reflexivity | ] ].
  rewrite He, Hk0. intros x Hx. exact Hx.
Qed.

Lemma uc_result_call_wraps (K : finite_map kshape) (ap0 ap ap' : apply_expr)
    (cs : code_shape) (k : continuation) (sh : list pshape)
    (KS : list flambda_kind) (wc : continuation) (cps : list variable)
    (cmd : handler_metadata) (cargs : list simple) (w : continuation)
    (wps : list variable) (wmd : handler_metadata) (wargs : list simple) :
  call_rel unboxed dead CSf ap ap' cs -> exn_ok K (ap_exn_continuation ap) ->
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
  split; [ exact (uc_call_wraps unboxed dead CSf K ap ap' cs k sh KS wc cps
                    cmd cargs w wps wmd wargs Hcall Hexn Hwc Hw Hwce Hwe Hlk
                    Hcmd Hok) | ].
  pose proof Hcall as (id & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & He' &
                       _).
  rewrite (fc_wrap wc) by (try exact Hwc; intros H; apply Hwce;
                           symmetry; exact H).
  rewrite (fc_wrap w) by (try exact Hw; rewrite He'; intros H; apply Hwe;
                          symmetry; exact H).
  rewrite He'. split; [ reflexivity | split; [ reflexivity | ] ].
  rewrite He. cbn [free_continuations_of_expr]. rewrite Hk0.
  intros x Hx. exact Hx.
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

(* The arguments of a call of code that keeps its calling convention:
   those that the callee uses as they are, the others poison. *)
Lemma args_rel2_mask (en : rebuild_env) (args : list simple) :
  ∀ (mask : list bool) (kinds : list flambda_kind),
    length args = length kinds ->
    forallb2
      (fun keep a =>
         negb keep
         || match a with
            | Var z => s_has_usage sol (Node_variable z)
            | Symbol _ | Const _ => true
            end)
      mask args
    = true ->
    forallb (plain_simple unboxed) args = true ->
    args_rel2 unboxed dead (map (fun _ => Psh_plain) kinds)
      (mask_arguments en mask args) (mask_arguments en mask args).
Proof.
  induction args as [ | a args IH]; intros [ | b mask] [ | k kinds] Hlen
    Hused Hplain; simpl in Hlen, Hused; try discriminate Hlen;
    try discriminate Hused.
  - apply args_rel2_beyond. reflexivity.
  - cbn [forallb] in Hplain. apply andb_true_iff in Hplain as [Ha Hplain].
    apply andb_true_iff in Hused as [Hb Hused].
    cbn [map mask_arguments]. apply args_rel2_plain;
      [ | exact (IH mask kinds ltac:(lia) Hused Hplain) ].
    destruct b; cbn [negb orb] in Hb.
    + destruct a as [z | sym | c]; [ | exact Ha | reflexivity ].
      apply plain2_iff. split; [ exact Ha | ].
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

Lemma make_apply_wrapper_kept (en : rebuild_env) (ap : apply_expr) :
  lookup (ap_continuation ap) (env_conts en) = Some [true]
  ∨ lookup (ap_continuation ap) (env_conts en) = None ->
  make_apply_wrapper ff en ap = Apply ap.
Proof.
  intros [H | H]; unfold make_apply_wrapper; rewrite H; reflexivity.
Qed.

Lemma make_apply_wrapper_deleted (en : rebuild_env) (ap : apply_expr)
    (kind : flambda_kind) :
  lookup (ap_continuation ap) (env_conts en) = Some [false] ->
  unarize (ap_return_arity ap) = [kind] ->
  make_apply_wrapper ff en ap
  = wrap ff [ff] (make_handler_metadata [kind]) (ap_continuation ap) []
      (Apply (with_continuation ap ff)).
Proof.
  intros H Hr. unfold make_apply_wrapper. rewrite H. cbn [forallb].
  rewrite Hr. reflexivity.
Qed.

(* A call of code that keeps its calling convention, as the two rebuilds
   make it: the first by FlambdaReaperRebuild.rebuild_apply, the second as
   a call that gives the callee its arguments and expects its results as
   they are, with a wrapper for the results that the continuation does
   not take as they are. *)
Lemma call_rebuilt_nc (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (blocked : list continuation)
    (ap : apply_expr) :
  cinv C blocked en enu ->
  call_ok F sol cannot_change code_deps unit_code_ids codes unboxed
    (env_kinds en) C ap
  = true ->
  (∀ z, z ∈ variables_in_apply ap -> fresh z) ->
  ap_exn_continuation ap <> ff -> ap_exn_continuation ap <> ffu ->
  apply_change_u F sol cannot_change code_deps unit_code_ids codes unboxed ap
  = Not_changing_u ->
  uc_result (K_of C) (free_continuations_of_expr (Apply ap))
    (cc_rebuild en (Apply ap)) (u_rebuild enu (Apply ap)).
Proof.
  intros Hci Hok Hfresh Hff Hffu Hac.
  destruct ap as [callee ckind args k exn extra AA RA].
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity] in *.
  set (ap := {| ap_callee := callee; ap_call_kind := ckind; ap_args := args;
                ap_continuation := k; ap_exn_continuation := exn;
                ap_exn_extra_args := extra; ap_args_arity := AA;
                ap_return_arity := RA |}) in *.
  unfold call_ok in Hok. cbn [ap_call_kind ap_callee ap ap_args
    ap_continuation ap_exn_continuation ap_exn_extra_args ap_args_arity
    ap_return_arity] in Hok.
  destruct ckind as [id | ]; [ | discriminate Hok ].
  destruct callee as [f | sym | cst]; try discriminate Hok.
  destruct (code_id_actually_called F sol ap) as [id' | ] eqn:Hcalled;
    [ | discriminate Hok ].
  apply andb_true_iff in Hok as [Hok Hrest].
  apply andb_true_iff in Hok as [Hok Hfused].
  apply andb_true_iff in Hok as [Hid Hfplain].
  apply Pos.eqb_eq in Hid. subst id'.
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
  apply andb_true_iff in Hrest as [Hrest Hextra].
  apply andb_true_iff in Hrest as [Hrest Hargs].
  apply andb_true_iff in Hrest as [Hrest Hrok].
  apply andb_true_iff in Hrest as [Hrest Hdok].
  apply andb_true_iff in Hrest as [Hrest Hrlen].
  apply andb_true_iff in Hrest as [Hrest Hdlen].
  apply andb_true_iff in Hrest as [Hrest Halen].
  apply andb_true_iff in Hrest as [Hrest Hmcd].
  apply andb_true_iff in Hrest as [Hrest Hres1].
  apply andb_true_iff in Hrest as [Hrest Hpar1].
  apply andb_true_iff in Hrest as [Haa Hra].
  apply arity_eqb_eq in Haa, Hra. subst AA RA.
  apply Nat.eqb_eq in Hres1, Hpar1, Halen, Hdlen, Hrlen.
  destruct extra as [ | ? ? ]; [ | discriminate Hextra ].
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
  destruct (cm_is_tupled md
            || existsb is_unbox
                 (fixed_arity_decisions sol unboxed (cd_return cd)
                    (unarize (cm_result_arity md))))
    eqn:Htup; [ discriminate Hpd | ].
  apply orb_false_iff in Htup as [_ Hnu].
  injection Hpd as Hpds Hrds. subst pds rds.
  destruct (cm_params_arity md) as [ | c [ | ? ? ]] eqn:Hpa;
    try discriminate Hpar1.
  destruct (unarize (cm_result_arity md)) as [ | kr [ | ? ? ]] eqn:Hua;
    try discriminate Hres1.
  rewrite !unarize_one in *.
  set (kinds := unarize_component c) in *.
  (* The one result, kept by the callee, used or not. *)
  unfold fixed_arity_decisions in Hnu, Hrlen, Hrok, Hret, Hpd'.
  destruct (cd_return cd) as [ | n_r ?] eqn:Hcr; [ discriminate Hrlen | ].
  cbn [combine map] in Hnu, Hrlen, Hrok, Hret, Hpd'.
  rewrite !combine_nil in *. cbn [map pdescs_of] in *.
  destruct (unboxed_of_node unboxed n_r) as [fr | ] eqn:Hnr;
    [ discriminate Hnu | ].
  destruct (lookup k C) as [[kps [ | ]] | ] eqn:HCk; try discriminate Hret.
  assert (Hkb : among k blocked = false).
  { destruct (among k blocked) eqn:E; [ | reflexivity ].
    destruct (ci_blocked _ _ _ _ Hci k E) as (ps0 & Hl).
    rewrite HCk in Hl. discriminate Hl. }
  destruct kps as [ | kp [ | kp2 kps]]; cbn [forallb2] in Hret;
    [ discriminate Hret | | rewrite andb_false_r in Hret; discriminate Hret ].
  rewrite andb_true_r in Hret.
  assert (Hfx : fields_of unboxed f = None).
  { cbn in Hfplain. apply negb_true_iff in Hfplain.
    unfold Keyed.is_bound in Hfplain. unfold fields_of.
    destruct (Keyed.lookup _ _ _); [ discriminate Hfplain | reflexivity ]. }
  assert (Hdead_eq :
    simple_is_dead sol (ue_env enu) (Var f)
    || existsb (simple_is_dead sol (ue_env enu)) args
    = simple_is_dead sol en (Var f) || existsb (simple_is_dead sol en) args).
  { f_equal; [ apply simple_is_dead_kinds, (ci_kinds _ _ _ _ Hci) | ].
    clear -Hci. induction args as [ | a args IH]; [ reflexivity | ].
    simpl. rewrite IH, (simple_is_dead_kinds _ _ a (ci_kinds _ _ _ _ Hci)).
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
             {| ap_callee := Var f; ap_call_kind := Direct id;
                ap_args := args; ap_continuation := k;
                ap_exn_continuation := exn; ap_exn_extra_args := [];
                ap_args_arity := [c];
                ap_return_arity := cm_result_arity md |}
             id eq_refl Hcalled).
  rewrite Hua.
  assert (Hexx : ∀ en0 : rebuild_env,
            rebuild_exn_extra_args sol en0
              {| ap_callee := Var f; ap_call_kind := Direct id;
                 ap_args := args; ap_continuation := k;
                 ap_exn_continuation := exn; ap_exn_extra_args := [];
                 ap_args_arity := [c];
                 ap_return_arity := cm_result_arity md |}
            = []).
  { intros en0. unfold rebuild_exn_extra_args.
    cbn [ap_exn_extra_args ap_exn_continuation map].
    destruct (lookup exn (env_conts en0)); reflexivity. }
  rewrite Hexx. cbn [rebuild_exn_extra_args_u get_extra_args ap_exn_extra_args].
  (* The arguments, the same in both runs. *)
  unfold rebuild_arguments.
  cbn [ap_callee ap_call_kind ap_args node_of_simple].
  unfold kept_args_ok in Hkept.
  apply andb_true_iff in Hkept as [Hmeq Hmused].
  apply forallb2_eqb_eq in Hmeq. rewrite <- Hmeq.
  rewrite (mask_arguments_kinds (ue_env enu) en args (ci_kinds _ _ _ _ Hci)).
  set (m := arguments_used_by_known_arity_call F sol (Node_variable f) args)
    in *.
  set (A := mask_arguments en m args).
  rewrite length_map in Halen.
  assert (Hplain : forallb (plain_simple unboxed) args = true).
  { apply strong_simples2; [ exact (args_ok2_keep _ args kinds Halen Hargs) | ].
    intros z Hz. apply Hfresh. unfold variables_in_apply.
    cbn [ap_callee ap_args variables_in_simple app]. right.
    apply in_or_app. left. exact Hz. }
  set (dr := if is_node_used sol n_r kr then Decided (Keep kr)
             else Keep_unused kr) in *.
  set (cs := Code_shape
               (flat_map pshape_of
                  (map (fun kind => Pdesc (Decided (Keep kind)) true kind)
                     kinds))
               (flat_map pshape_of [Pdesc dr true kr])
               (flat_map kept_kind [Pdesc dr true kr]) [c]
               (cm_result_arity md)).
  assert (HCS : CSf id = Some cs).
  { unfold CSf. rewrite Hcodes, Hnc, Hpd', Hpa. reflexivity. }
  set (ap1 := {| ap_callee := Var f; ap_call_kind := Direct id; ap_args := A;
                 ap_continuation := k; ap_exn_continuation := exn;
                 ap_exn_extra_args := []; ap_args_arity := [c];
                 ap_return_arity := cm_result_arity md |}).
  assert (Hcall : call_rel unboxed dead CSf ap1 ap1 cs).
  { exists id. split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ exact HCS | ].
    split; [ reflexivity | ].
    split; [ apply (plain2_of f Value Hfx); [ | reflexivity | exact Hfused ];
             apply Hfresh; left; reflexivity | ].
    split; [ cbn [cs cs_params ap1 ap_args]; rewrite plain_shapes;
             exact (args_rel2_mask en args m kinds Halen Hmused Hplain) | ].
    split; [ cbn [cs cs_params]; rewrite plain_shapes;
             apply plain_shapes_no_dead | ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | ].
    split; [ cbn [cs cs_params ap1 ap_args_arity cs_params_arity];
             rewrite plain_shapes, unarize_one, plain_shapes_kinds;
             reflexivity | ].
    split; [ reflexivity | split; reflexivity ]. }
  assert (Hexn_ok : exn_ok (K_of C) exn) by exact (exn_cont_ok_K C exn Hexn).
  assert (HKk : lookup k (K_of C)
                = Some (Kshape (flat_map pshape_of [kp])
                          (flat_map kept_kind [kp]) true))
    by (rewrite lookup_K_of, HCk; reflexivity).
  pose proof (ci_ok _ _ _ _ Hci k [kp] true HCk) as Hkp_ok.
  cbn [forallb] in Hkp_ok. rewrite andb_true_r in Hkp_ok.
  unfold result_ok in Hret. apply andb_true_iff in Hret as [Hkind Hret].
  apply kind_eqb_eq in Hkind.
  assert (Hmask : lookup k (env_conts en) = Some [pd_keep kp]
                  ∨ lookup k (env_conts en) = None ∧ pd_keep kp = true).
  { destruct (ci_mask _ _ _ _ Hci k Hkb) as [Hm | (Hm & ps & h & HC & Hall)].
    - left. rewrite Hm, HCk. reflexivity.
    - right. split; [ exact Hm | ]. rewrite HCk in HC.
      injection HC as <- _. cbn in Hall. rewrite andb_true_r in Hall.
      exact Hall. }
  unfold make_apply_wrapper_u.
  change (ap_continuation _) with k.
  rewrite (ci_dec _ _ _ _ Hci k Hkb), HCk.
  cbn [option_map fst map freshen_decisions wrapper_params flat_map app].
  destruct kp as [dk mk kk]. cbn [pd_decision pd_keep pd_kind] in *.
  subst kk.
  set (ap0 := Apply_expr (Var f) (Direct id) args k exn [] [c]
                (cm_result_arity md)).
  assert (Hffe : ff <> exn) by (intros H; exact (Hff (eq_sym H))).
  assert (Hffue : ffu <> exn) by (intros H; exact (Hffu (eq_sym H))).
  assert (Hsh : flat_map pshape_of [Pdesc dr true kr] = [Psh_plain]
                ∨ flat_map pshape_of [Pdesc dr true kr] = [Psh_dead kr]).
  { unfold dr. destruct (is_node_used sol n_r kr); [ left | right ];
      reflexivity. }
  assert (Hfit : Forall2 fits (flat_map pshape_of [Pdesc dr true kr])
                   [Psh_dead kr]).
  { destruct Hsh as [-> | ->]; repeat constructor. }
  unfold pdesc_ok in Hkp_ok. cbn [pd_kind pd_decision pd_keep] in Hkp_ok.
  apply andb_true_iff in Hkp_ok as [_ Hkp_ok].
  destruct dk as [[kk' | ] | fk | kk'].
  - (* Kept by the continuation, so by the callee. *)
    apply andb_true_iff in Hkp_ok as [Hmk Hkk]. subst mk.
    apply kind_eqb_eq in Hkk. subst kk'.
    rewrite (make_apply_wrapper_kept en ap1)
      by (destruct Hmask as [Hm | [Hm _]]; [ left | right ]; exact Hm).
    cbn [wrapper_args_u args_are_params map fst]. rewrite Pos.eqb_refl.
    cbn [andb].
    unfold dr in *. destruct (is_node_used sol n_r kr); [ | discriminate Hret ].
    apply (uc_result_call (K_of C) ap0 ap1 ap1 _ k _ Hcall Hexn_ok
             eq_refl eq_refl eq_refl eq_refl HKk).
    apply transport_refl.
  - (* Deleted by the continuation: wrappers in both runs. *)
    apply negb_true_iff in Hkp_ok. subst mk.
    destruct Hmask as [Hm | [_ Hm]]; [ | discriminate Hm ].
    rewrite (make_apply_wrapper_deleted en ap1 kr Hm Hua).
    cbn [wrapper_args_u args_are_params map fst].
    eapply (uc_result_call_wraps (K_of C) ap0 _ _ _ k
             (flat_map pshape_of [Pdesc (Decided Delete) false kr]) [] ff
             _ _ _ ffu _ _ _ (call_rel_with _ _ _ ff ffu Hcall) Hexn_ok
             eq_refl eq_refl Hffe Hffue eq_refl eq_refl HKk);
      [ reflexivity | ].
    intros vs vs' Hvr Hck.
    apply kinds_eqb_eq in Hck. cbn in Hck.
    destruct (vrel_one_kept _ vs vs' kr Hsh Hvr Hck)
      as (v & v' & -> & -> & Hkv & _).
    split; [ cbn; rewrite Hkv; apply kinds_eqb_refl | ].
    intros cv wv hv Hb. exists ((ffu, v') :: wv). split; [ reflexivity | ].
    intros ws Hws0. cbn in Hws0. injection Hws0 as <-.
    exists []. split; [ reflexivity | constructor ].
  - unfold dr in Hret. destruct (is_node_used sol n_r kr); discriminate Hret.
  - (* Kept with no use by the continuation: the fix and its variant. *)
    apply andb_true_iff in Hkp_ok as [Hmk Hkk]. subst mk.
    apply kind_eqb_eq in Hkk. subst kk'.
    rewrite (make_apply_wrapper_kept en ap1)
      by (destruct Hmask as [Hm | [Hm _]]; [ left | right ]; exact Hm).
    cbn [wrapper_args_u].
    destruct tail; cbn [args_are_params map fst].
    + rewrite Pos.eqb_refl. cbn [andb].
      apply (uc_result_call (K_of C) ap0 ap1 ap1 _ k _ Hcall Hexn_ok
               eq_refl eq_refl eq_refl eq_refl HKk).
      apply transport_fits. exact Hfit.
    + eapply (uc_result_call_wrap (K_of C) ap0 ap1
                (with_continuation ap1 ffu) _ k _ ffu _ _ _
                (call_rel_with _ _ _ k ffu Hcall) Hexn_ok eq_refl eq_refl
                Hffue eq_refl eq_refl HKk).
      intros vs vs' Hvr Hck.
      destruct (vrel_one_kept _ vs vs' kr Hsh Hvr Hck)
        as (v & v' & -> & -> & Hkv & _).
      split; [ cbn; rewrite Hkv; apply kinds_eqb_refl | ].
      intros wv. exists ((ffu, v') :: wv), [Value_const (Poison kr)].
      split; [ reflexivity | split; [ reflexivity | ] ].
      apply vrel_dead; [ right; reflexivity | constructor ].
Qed.

(* A call, as the two rebuilds make it. *)
Lemma call_rebuilt (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list pdesc * bool)) (blocked : list continuation)
    (ap : apply_expr) :
  cinv C blocked en enu ->
  call_ok F sol cannot_change code_deps unit_code_ids codes unboxed
    (env_kinds en) C ap
  = true ->
  (∀ z, z ∈ variables_in_apply ap -> fresh z) ->
  ap_exn_continuation ap <> ff -> ap_exn_continuation ap <> ffu ->
  uc_result (K_of C) (free_continuations_of_expr (Apply ap))
    (cc_rebuild en (Apply ap)) (u_rebuild enu (Apply ap)).
Proof.
  intros Hci Hok Hfresh Hff Hffu.
  destruct (apply_change_u F sol cannot_change code_deps unit_code_ids codes
              unboxed ap)
    as [ | pd0 rd0] eqn:Hac;
    [ exact (call_rebuilt_nc en enu C blocked ap Hci Hok Hfresh Hff Hffu Hac)
    | ].
  destruct ap as [callee ckind args k exn extra AA RA].
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity] in *.
  set (ap := {| ap_callee := callee; ap_call_kind := ckind; ap_args := args;
                ap_continuation := k; ap_exn_continuation := exn;
                ap_exn_extra_args := extra; ap_args_arity := AA;
                ap_return_arity := RA |}) in *.
  unfold call_ok in Hok. cbn [ap_call_kind ap_callee ap ap_args
    ap_continuation ap_exn_continuation ap_exn_extra_args ap_args_arity
    ap_return_arity] in Hok.
  destruct ckind as [id | ]; [ | discriminate Hok ].
  destruct callee as [f | sym | cst]; try discriminate Hok.
  destruct (code_id_actually_called F sol ap) as [id' | ] eqn:Hcalled;
    [ | discriminate Hok ].
  apply andb_true_iff in Hok as [Hok Hrest].
  apply andb_true_iff in Hok as [Hok Hfused].
  apply andb_true_iff in Hok as [Hid Hfplain].
  apply Pos.eqb_eq in Hid. subst id'.
  destruct (lookup id codes) as [md | ] eqn:Hcodes; [ | discriminate Hrest ].
  unfold apply_change_u in Hac. rewrite Hcalled, Hcodes in Hac.
  destruct (code_pdescs sol cannot_change code_deps unit_code_ids unboxed id
              md)
    as [[pds rds] | ] eqn:Hpd; [ | discriminate Hrest ].
  apply andb_true_iff in Hrest as [Hrest Hret].
  apply andb_true_iff in Hrest as [Hrest _].
  apply andb_true_iff in Hrest as [Hrest Hexn].
  apply andb_true_iff in Hrest as [Hrest Hextra].
  apply andb_true_iff in Hrest as [Hrest Hargs].
  apply andb_true_iff in Hrest as [Hrest Hrok].
  apply andb_true_iff in Hrest as [Hrest Hdok].
  apply andb_true_iff in Hrest as [Hrest Hrlen].
  apply andb_true_iff in Hrest as [Hrest Hdlen].
  apply andb_true_iff in Hrest as [Hrest Halen].
  apply andb_true_iff in Hrest as [Hrest Hmcd].
  apply andb_true_iff in Hrest as [Hrest Hres1].
  apply andb_true_iff in Hrest as [Hrest Hpar1].
  apply andb_true_iff in Hrest as [Haa Hra].
  apply arity_eqb_eq in Haa, Hra. subst AA RA.
  apply Nat.eqb_eq in Hres1, Hpar1, Halen, Hdlen, Hrlen.
  destruct extra as [ | ? ? ]; [ | discriminate Hextra ].
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
  destruct (cm_params_arity md) as [ | c [ | ? ? ]] eqn:Hpa;
    try discriminate Hpar1.
  destruct (unarize (cm_result_arity md)) as [ | kr [ | ? ? ]] eqn:Hua;
    try discriminate Hres1.
  rewrite !unarize_one in *.
  set (kinds := unarize_component c) in *.
  assert (Hlen_pd : length pd = length kinds ∧ length pd_cc = length kinds).
  { rewrite pdescs_of_length, length_map in Hdlen.
    rewrite <- Hpd, <- Hpd_cc, decisions_u_length, decisions_of_length'.
    rewrite <- (Forall2_length Hlink) in Hdlen.
    rewrite <- Hpd, decisions_u_length in Hdlen.
    rewrite length_combine in Hdlen |- *. lia. }
  destruct Hlen_pd as [Hlpd Hlpd_cc].
  (* The one result. *)
  subst rd rd_cc.
  destruct (cd_return cd) as [ | n_r ?] eqn:Hcr; [ discriminate Hrlen | ].
  unfold decisions_u, decisions_of in *. cbn [combine] in *.
  rewrite !combine_nil in *. cbn [map pdescs_of] in *.
  set (dr := match unboxed_of_node unboxed n_r with
             | Some fields => Unbox fields
             | None => Decided (if is_node_used sol n_r kr then Keep kr
                                else Delete)
             end) in *.
  set (drcc := if is_node_used sol n_r kr then Keep kr else Delete) in *.
  set (r := Pdesc dr (is_kept drcc) kr) in *.
  cbn [forallb] in Hrok. rewrite andb_true_r in Hrok.
  destruct (lookup k C) as [[kps [ | ]] | ] eqn:HCk; try discriminate Hret.
  assert (Hkb : among k blocked = false).
  { destruct (among k blocked) eqn:E; [ | reflexivity ].
    destruct (ci_blocked _ _ _ _ Hci k E) as (ps0 & Hl).
    rewrite HCk in Hl. discriminate Hl. }
  destruct kps as [ | kp [ | kp2 kps]]; cbn [forallb2] in Hret;
    [ discriminate Hret | | rewrite andb_false_r in Hret; discriminate Hret ].
  rewrite andb_true_r in Hret.
  unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
  assert (Hfx : fields_of unboxed f = None).
  { cbn in Hfplain. apply negb_true_iff in Hfplain.
    unfold Keyed.is_bound in Hfplain. unfold fields_of.
    destruct (Keyed.lookup _ _ _); [ discriminate Hfplain | reflexivity ]. }
  assert (Hdead_eq :
    simple_is_dead sol (ue_env enu) (Var f)
    || existsb (simple_is_dead sol (ue_env enu)) args
    = simple_is_dead sol en (Var f) || existsb (simple_is_dead sol en) args).
  { f_equal; [ apply simple_is_dead_kinds, (ci_kinds _ _ _ _ Hci) | ].
    clear -Hci. induction args as [ | a args IH]; [ reflexivity | ].
    simpl. rewrite IH, (simple_is_dead_kinds _ _ a (ci_kinds _ _ _ _ Hci)).
    reflexivity. }
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
             {| ap_callee := Var f; ap_call_kind := Direct id;
                ap_args := args; ap_continuation := k;
                ap_exn_continuation := exn; ap_exn_extra_args := [];
                ap_args_arity := [c];
                ap_return_arity := cm_result_arity md |}
             id eq_refl Hcalled).
  assert (Hexx : ∀ en0 : rebuild_env,
            rebuild_exn_extra_args sol en0
              {| ap_callee := Var f; ap_call_kind := Direct id;
                 ap_args := args; ap_continuation := k;
                 ap_exn_continuation := exn; ap_exn_extra_args := [];
                 ap_args_arity := [c];
                 ap_return_arity := cm_result_arity md |}
            = []).
  { intros en0. unfold rebuild_exn_extra_args.
    cbn [ap_exn_extra_args ap_exn_continuation map].
    destruct (lookup exn (env_conts en0)); reflexivity. }
  rewrite Hexx. cbn [rebuild_exn_extra_args_u get_extra_args ap_exn_extra_args].
  assert (Hal : length args <= length kinds) by lia.
  assert (Hpl : length pd <= length kinds) by lia.
  assert (Hpcl : length pd_cc <= length kinds) by lia.
  rewrite (group_by_one c args Hal), (group_by_one c pd Hpl),
    (group_by_one c pd_cc Hpcl).
  cbn [combine map].
  assert (Hkok : ∀ x, kind_ok (kind_of_variable en x) = true)
    by exact (ci_kinds_ok _ _ _ _ Hci).
  destruct (call_args_rel en enu args (ci_kinds _ _ _ _ Hci) Hkok)
    with (pd := pd) (pd_cc := pd_cc) (kinds := kinds)
    as (l_u & Hgu & Hargs_rel & Hk1 & Hk2).
  { intros z Hz. apply Hfresh. unfold variables_in_apply.
    cbn [ap_callee ap_args variables_in_simple app]. right.
    apply in_or_app. left. exact Hz. }
  { exact Hlink. }
  { exact Halen. }
  { exact Hlpd. }
  { exact Hlpd_cc. }
  { exact Hdok. }
  { exact Hargs. }
  rewrite Hgu. cbn [all_some fold_right prepend_first app].
  change (unboxed_of unboxed (Var f)) with (fields_of unboxed f).
  rewrite Hfx.
  set (l_cc := get_args_with_kinds sol en pd_cc args) in *.
  rewrite Hua.
  match goal with
  | |- uc_result _ _ (make_apply_wrapper_cc _ _ ?A _)
         (make_apply_wrapper_u _ _ _ ?B _) =>
    set (ap_cc := A); set (ap_u := B)
  end.
  set (cs := Code_shape (flat_map pshape_of
                           (pdescs_of pd (map is_kept pd_cc) kinds))
               (flat_map pshape_of [r]) (flat_map kept_kind [r])
               (arity_of_grouped_decisions_u (prepend_first [Decided Delete]
                                                [pd]))
               (arity_of_decisions_u [dr])).
  assert (HCS : CSf id = Some cs).
  { unfold CSf. rewrite Hcodes, Hchu', Hpd', Hpa.
    rewrite (group_by_one c pd Hpl). reflexivity. }
  assert (Hcall : call_rel unboxed dead CSf ap_cc ap_u cs).
  { exists id. split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ exact HCS | ].
    split; [ reflexivity | ].
    split; [ apply (plain2_of f Value Hfx); [ | reflexivity | exact Hfused ];
             apply Hfresh; left; reflexivity | ].
    split; [ cbn [ap_args ap_cc ap_u flat_map]; rewrite !app_nil_r;
             exact Hargs_rel | ].
    split; [ exact (no_dead_code pd pd_cc _ kinds Hlink) | ].
    split.
    { cbn [ap_args_arity ap_u cs cs_params_arity]. unfold arity_of_groups.
      cbn [map prepend_first app]. unfold arity_of_grouped_decisions_u.
      cbn [map]. f_equal. f_equal. rewrite singletons_snd, Hk1.
      unfold decisions_kinds. reflexivity. }
    split; [ reflexivity | ].
    split; [ reflexivity | ].
    split.
    { cbn [cs cs_params_arity ap_args_arity ap_cc]. unfold arity_of_groups.
      unfold arity_of_grouped_decisions_u. cbn [map prepend_first app].
      rewrite singletons_snd, !unarize_singletons.
      unfold decisions_kinds in Hk2 |- *. cbn [flat_map decision_kinds
        decision_params map app]. exact (eq_sym Hk2). }
    split; [ reflexivity | split; reflexivity ]. }
  assert (Hexn_ok : exn_ok (K_of C) exn) by exact (exn_cont_ok_K C exn Hexn).
  assert (HKk : lookup k (K_of C)
                = Some (Kshape (flat_map pshape_of [kp])
                          (flat_map kept_kind [kp]) true))
    by (rewrite lookup_K_of, HCk; reflexivity).
  pose proof (ci_ok _ _ _ _ Hci k [kp] true HCk) as Hkp_ok.
  cbn [forallb] in Hkp_ok. rewrite andb_true_r in Hkp_ok.
  unfold result_ok in Hret. apply andb_true_iff in Hret as [Hkind Hret].
  apply kind_eqb_eq in Hkind.
  assert (Happ : apply_decisions en k 1 = [pd_keep kp]).
  { unfold apply_decisions.
    destruct (ci_mask _ _ _ _ Hci k Hkb) as [Hm | (Hm & ps & h & HC & Hall)];
      rewrite Hm; [ rewrite HCk; reflexivity | ].
    rewrite HCk in HC. injection HC as <- _. cbn in Hall.
    rewrite andb_true_r in Hall. rewrite Hall. reflexivity. }
  unfold make_apply_wrapper_cc, make_apply_wrapper_u.
  cbn [length combine]. cbv zeta.
  change (ap_continuation ap_cc) with k.
  change (ap_continuation ap_u) with k.
  rewrite Happ, (ci_dec _ _ _ _ Hci k Hkb), HCk.
  cbn [option_map fst map].
  destruct kp as [dk mk kk]. cbn [pd_decision pd_keep pd_kind] in *.
  subst kk.
  set (ap0 := Apply_expr (Var f) (Direct id) args k exn [] [c]
                (cm_result_arity md)).
  assert (Hffe : ff <> exn) by (intros H; exact (Hff (eq_sym H))).
  assert (Hffue : ffu <> exn) by (intros H; exact (Hffu (eq_sym H))).
  unfold cs in HCS, Hcall. unfold r, dr, drcc in *. clear cs r dr drcc.
  unfold pdesc_ok in Hrok, Hkp_ok.
  cbn [pd_kind pd_decision pd_keep] in Hrok, Hkp_ok.
  apply andb_true_iff in Hrok as [Hkr Hrok].
  apply andb_true_iff in Hkp_ok as [_ Hkp_ok].
  destruct (unboxed_of_node unboxed n_r) as [fr | ] eqn:Hnr;
    destruct (is_node_used sol n_r kr) eqn:Hnu.
  - (* A result that the callee unboxes, kept. *)
    apply andb_true_iff in Hrok as [Hv _]. apply kind_eqb_eq in Hv.
    subst kr.
    destruct (freshen_unbox fr Value ffu)
      as (fr' & Hfresh' & Hpaths & Hnd & Hren).
    rewrite Hfresh'. cbn [wrapper_params flat_map app].
    destruct dk as [[kk' | ] | fk | kk'].
    + discriminate Hret.
    + (* Deleted by the continuation: wrappers in both runs. *)
      apply negb_true_iff in Hkp_ok. subst mk.
      cbn [is_kept forallb combine Bool.eqb andb kept_kinds flat_map
           wrapper_args_u wrapper_args app].
      assert (Hleaves : fields_leaves fr' <> []).
      { intros Hnil. apply negb_true_iff in Hret.
        rewrite andb_true_r in Hret. unfold no_leaves in Hret.
        assert (Hp' : fields_paths fr' = [])
          by (pose proof (Forall2_length (fields_leaves_paths fr')) as Hl;
              rewrite Hnil in Hl; destruct (fields_paths fr');
              [ reflexivity | discriminate Hl ]).
        rewrite Hpaths in Hp'. apply leaves_of_paths_nil in Hp'.
        rewrite Hp' in Hret. discriminate Hret. }
      destruct (fields_leaves fr') as [ | [x0 k0] l0] eqn:Hl0;
        [ exfalso; exact (Hleaves eq_refl) | ].
      cbn [args_are_params map fst].
      eapply (uc_result_call_wraps (K_of C) ap0 _ _ _ k
               (flat_map pshape_of [Pdesc (Decided Delete) false Value]) [] ff
               _ _ _ ffu _ _ _ (call_rel_with _ _ _ ff ffu Hcall) Hexn_ok
               eq_refl eq_refl Hffe Hffue eq_refl eq_refl HKk);
        [ reflexivity | ].
      intros vs vs' Hvr Hck.
      apply kinds_eqb_eq in Hck. cbn in Hck.
      destruct (vrel_one_unboxed fr vs vs' Value Hvr Hck) as (v & -> & Hws).
      split.
      { apply kinds_eqb_of. cbn [hm_params_kinds make_handler_metadata].
        rewrite app_nil_r, <- Hl0.
        exact (unbox_kinds fr fr' v vs' Hpaths Hws). }
      intros cv wv hv Hb.
      destruct (bind_unbox_wrapper fr fr' v vs' wv Hpaths Hnd Hws)
        as (hv' & Hb' & _).
      exists hv'. split; [ unfold leaf_variables in Hb'; rewrite Hl0 in Hb';
                           rewrite app_nil_r; exact Hb' | ].
      intros ws Hws0. cbn in Hws0. injection Hws0 as <-.
      exists []. split; [ reflexivity | constructor ].
    + (* Unboxed by both. *)
      apply andb_true_iff in Hret as [Halike Hret].
      apply andb_true_iff in Hkp_ok as [_ Hkpk].
      unfold fields_alike in Halike.
      destruct (unboxed_field_pairs Value (Unboxed fk) (Unboxed fr))
        as [pairs0 | ] eqn:Hp0; [ | discriminate Halike ].
      destruct (pairs_rename (Unboxed fk) Value (Unboxed fr) ffu pairs0 Hp0)
        as (pairs & Hp). rewrite Hren in Hp.
      cbn [wrapper_args_u]. rewrite Hp.
      destruct mk.
      * (* Kept by both. *)
        cbn [is_kept forallb combine Bool.eqb andb].
        assert (Hsem : ∀ (v : value) (ws : list value) (wv : env),
                  fields_vals fr v ws ->
                  ∃ (hv : env) (ws'' : list value),
                    bind_params (map fst (fields_leaves fr' ++ [])) ws wv
                    = Some hv
                    ∧ eval_simples hv
                        (map (fun '(_, x, _) => Var x) pairs ++ [])
                      = Some ws''
                    ∧ fields_vals fk v ws'').
        { intros v ws wv Hws.
          destruct (bind_unbox_wrapper fr fr' v ws wv Hpaths Hnd Hws)
            as (hv & Hb & Hhold).
          destruct (pairs_fields_vals hv fk fr' v pairs Hp Hhold)
            as (ws'' & Hev & Hfk).
          exists hv, ws''. rewrite !app_nil_r.
          split; [ exact Hb | split; [ exact Hev | exact Hfk ] ]. }
        destruct (args_are_params (map (fun '(_, x, _) => Var x) pairs ++ [])
                    (map fst (fields_leaves fr' ++ [])))
          eqn:Hap.
        -- apply (uc_result_call (K_of C) ap0 ap_cc ap_u _ k _ Hcall Hexn_ok
                    eq_refl eq_refl eq_refl eq_refl HKk).
           intros vs vs' Hvr Hck.
           destruct (vrel_one_unboxed fr vs vs' Value Hvr Hck)
             as (v & -> & Hws).
           destruct (Hsem v vs' [] Hws) as (hv & ws'' & Hb & Hev & Hfk).
           apply args_are_params_eq in Hap. rewrite Hap in Hev.
           rewrite (eval_bound_vars _ vs' [] hv) in Hev;
             [ | rewrite map_app, app_nil_r; exact Hnd | exact Hb ].
           injection Hev as <-.
           rewrite <- (app_nil_r vs').
           apply vrel_unboxed; [ exact Hfk | constructor ].
        -- eapply (uc_result_call_wrap (K_of C) ap0 ap_cc
                     (with_continuation ap_u ffu) _ k _ ffu _ _ _
                     (call_rel_with _ _ _ k ffu Hcall) Hexn_ok eq_refl eq_refl
                     Hffue eq_refl eq_refl HKk).
           intros vs vs' Hvr Hck.
           destruct (vrel_one_unboxed fr vs vs' Value Hvr Hck)
             as (v & -> & Hws).
           split.
           { apply kinds_eqb_of. cbn [hm_params_kinds make_handler_metadata].
             rewrite app_nil_r. exact (unbox_kinds fr fr' v vs' Hpaths Hws). }
           intros wv. destruct (Hsem v vs' wv Hws) as (hv & ws'' & Hb & Hev
                                                       & Hfk).
           exists hv, (ws'' ++ []). split; [ exact Hb | ].
           split; [ rewrite !app_nil_r in Hev; rewrite !app_nil_r;
                    exact Hev | ].
           apply vrel_unboxed; [ exact Hfk | constructor ].
      * (* Deleted by the continuation, kept by the callee. *)
        cbn [orb] in Hkpk.
        rewrite (pairs_no_leaves fk fr' pairs Hp (no_leaves_nil fk Hkpk)).
        cbn [is_kept forallb combine Bool.eqb andb kept_kinds flat_map
             wrapper_args app map].
        cbn [Bool.eqb orb] in Hret. rewrite orb_false_r in Hret.
        apply negb_true_iff in Hret.
        assert (Hleaves : fields_leaves fr' <> []).
        { intros Hnil.
          assert (Hp' : fields_paths fr' = [])
            by (pose proof (Forall2_length (fields_leaves_paths fr')) as Hl;
                rewrite Hnil in Hl; destruct (fields_paths fr');
                [ reflexivity | discriminate Hl ]).
          rewrite Hpaths in Hp'. apply leaves_of_paths_nil in Hp'.
          unfold no_leaves in Hret. rewrite Hp' in Hret. discriminate Hret. }
        destruct (fields_leaves fr') as [ | [x0 k0] l0] eqn:Hl0;
          [ exfalso; exact (Hleaves eq_refl) | ].
        cbn [args_are_params map fst].
        eapply (uc_result_call_wraps (K_of C) ap0 _ _ _ k
                 (flat_map pshape_of [Pdesc (Unbox fk) false Value]) [] ff
                 _ _ _ ffu _ _ _ (call_rel_with _ _ _ ff ffu Hcall) Hexn_ok
                 eq_refl eq_refl Hffe Hffue eq_refl eq_refl HKk);
          [ reflexivity | ].
        intros vs vs' Hvr Hck.
        apply kinds_eqb_eq in Hck. cbn in Hck.
        destruct (vrel_one_unboxed fr vs vs' Value Hvr Hck) as (v & -> & Hws).
        split.
        { apply kinds_eqb_of. cbn [hm_params_kinds make_handler_metadata].
          rewrite app_nil_r, <- Hl0.
          exact (unbox_kinds fr fr' v vs' Hpaths Hws). }
        intros cv wv hv Hb.
        destruct (bind_unbox_wrapper fr fr' v vs' wv Hpaths Hnd Hws)
          as (hv' & Hb' & _).
        exists hv'. split; [ unfold leaf_variables in Hb'; rewrite Hl0 in Hb';
                             rewrite app_nil_r; exact Hb' | ].
        intros ws Hws0. cbn in Hws0. injection Hws0 as <-.
        exists []. split; [ reflexivity | constructor ].
    + (* Kept with no use by the continuation. *)
      apply andb_true_iff in Hkp_ok as [Hmk Hkk]. subst mk.
      apply kind_eqb_eq in Hkk. subst kk'.
      cbn [is_kept forallb combine Bool.eqb andb wrapper_args_u app
           args_are_params].
      eapply (uc_result_call_wrap (K_of C) ap0 ap_cc
                (with_continuation ap_u ffu) _ k _ ffu _ _ _
                (call_rel_with _ _ _ k ffu Hcall) Hexn_ok eq_refl eq_refl
                Hffue eq_refl eq_refl HKk).
      intros vs vs' Hvr Hck.
      destruct (vrel_one_unboxed fr vs vs' Value Hvr Hck) as (v & -> & Hws).
      split.
      { apply kinds_eqb_of. cbn [hm_params_kinds make_handler_metadata].
        rewrite app_nil_r. exact (unbox_kinds fr fr' v vs' Hpaths Hws). }
      intros wv.
      destruct (bind_unbox_wrapper fr fr' v vs' wv Hpaths Hnd Hws)
        as (hv & Hb & _).
      exists hv, [Value_const (Poison Value)].
      split; [ rewrite app_nil_r; exact Hb | split; [ reflexivity | ] ].
      apply vrel_dead; [ right; reflexivity | constructor ].
  - (* A result that the callee unboxes, not kept. *)
    apply andb_true_iff in Hrok as [Hv Hnl]. apply kind_eqb_eq in Hv.
    subst kr. cbn [orb is_kept] in Hnl.
    destruct (freshen_unbox fr Value ffu)
      as (fr' & Hfresh' & Hpaths & Hnd & Hren).
    assert (Hl0 : fields_leaves fr' = []).
    { apply leaves_of_paths_nil. rewrite Hpaths.
      exact (no_leaves_paths fr Hnl). }
    rewrite Hfresh'. cbn [wrapper_params flat_map app]. rewrite Hl0.
    destruct dk as [[kk' | ] | fk | kk'].
    + discriminate Hret.
    + apply negb_true_iff in Hkp_ok. subst mk.
      cbn [is_kept forallb combine Bool.eqb andb wrapper_args_u app map
           args_are_params].
      apply (uc_result_call (K_of C) ap0 ap_cc ap_u _ k _ Hcall Hexn_ok
               eq_refl eq_refl eq_refl eq_refl HKk).
      apply transport_refl.
    + apply andb_true_iff in Hret as [Halike Hret].
      apply andb_true_iff in Hkp_ok as [_ Hkpk].
      rewrite Hnl in Hret. cbn [negb orb] in Hret.
      destruct mk; [ discriminate Hret | ]. cbn [orb] in Hkpk.
      unfold fields_alike in Halike.
      destruct (unboxed_field_pairs Value (Unboxed fk) (Unboxed fr))
        as [pairs0 | ] eqn:Hp0; [ | discriminate Halike ].
      destruct (pairs_rename (Unboxed fk) Value (Unboxed fr) ffu pairs0 Hp0)
        as (pairs & Hp). rewrite Hren in Hp.
      cbn [wrapper_args_u]. rewrite Hp.
      rewrite (pairs_no_leaves fk fr' pairs Hp (no_leaves_nil fk Hkpk)).
      cbn [is_kept forallb combine Bool.eqb andb app map args_are_params].
      apply (uc_result_call (K_of C) ap0 ap_cc ap_u _ k _ Hcall Hexn_ok
               eq_refl eq_refl eq_refl eq_refl HKk).
      apply transport_refl.
    + apply andb_true_iff in Hkp_ok as [Hmk Hkk]. subst mk.
      apply kind_eqb_eq in Hkk. subst kk'.
      cbn [is_kept forallb combine Bool.eqb andb kept_kinds flat_map
           wrapper_args wrapper_args_u app map length
           FlambdaFreshQuantifierSpec.names_from
           args_are_params].
      eapply (uc_result_call_wraps (K_of C) ap0 _ _ _ k
               (flat_map pshape_of [Pdesc (Keep_unused Value) true Value]) _
               ff _ _ _ ffu _ _ _ (call_rel_with _ _ _ ff ffu Hcall) Hexn_ok
               eq_refl eq_refl Hffe Hffue eq_refl eq_refl HKk);
        [ reflexivity | ].
      intros vs vs' Hvr Hck.
      apply vrel_none in Hvr. subst vs'.
      apply kinds_eqb_eq in Hck. destruct vs; [ | discriminate Hck ].
      split; [ reflexivity | ].
      intros cv wv hv Hb. cbn in Hb. injection Hb as <-.
      exists wv. split; [ reflexivity | ].
      intros ws Hws0. cbn in Hws0. injection Hws0 as <-.
      exists [Value_const (Poison Value)]. split; [ reflexivity | ].
      apply vrel_dead; [ left; reflexivity | constructor ].
  - (* A result that the callee keeps. *)
    cbn [freshen_decisions wrapper_params flat_map app].
    destruct dk as [[kk' | ] | fk | kk'].
    + apply andb_true_iff in Hkp_ok as [Hmk _]. subst mk.
      cbn [is_kept forallb combine Bool.eqb andb wrapper_args_u app map fst
           args_are_params]. rewrite Pos.eqb_refl. cbn [andb].
      apply (uc_result_call (K_of C) ap0 ap_cc ap_u _ k _ Hcall Hexn_ok
               eq_refl eq_refl eq_refl eq_refl HKk).
      apply transport_refl.
    + apply negb_true_iff in Hkp_ok. subst mk.
      cbn [is_kept forallb combine Bool.eqb andb kept_kinds flat_map
           wrapper_args wrapper_args_u app map fst length
           FlambdaFreshQuantifierSpec.names_from
           args_are_params].
      eapply (uc_result_call_wraps (K_of C) ap0 _ _ _ k
               (flat_map pshape_of [Pdesc (Decided Delete) false kr]) [] ff
               _ _ _ ffu _ _ _ (call_rel_with _ _ _ ff ffu Hcall) Hexn_ok
               eq_refl eq_refl Hffe Hffue eq_refl eq_refl HKk);
        [ reflexivity | ].
      intros vs vs' Hvr Hck.
      apply kinds_eqb_eq in Hck. cbn in Hck.
      destruct (vrel_one_plain vs vs' kr Hvr Hck) as (v & -> & ->).
      split; [ exact (proj2 (kinds_eqb_eq _ _) Hck) | ].
      intros cv wv hv Hb. exists ((ffu, v) :: wv). split; [ reflexivity | ].
      intros ws Hws0. cbn in Hws0. injection Hws0 as <-.
      exists []. split; [ reflexivity | constructor ].
    + discriminate Hret.
    + apply andb_true_iff in Hkp_ok as [Hmk Hkk]. subst mk.
      apply kind_eqb_eq in Hkk. subst kk'.
      (* The one case where the fix ([make_apply_wrapper_u]: poison) and
         the variant that passes the result through differ; the proof
         is that of either. *)
      cbn [is_kept forallb combine Bool.eqb andb wrapper_args_u app].
      destruct (args_are_params _ _) eqn:Hap.
      * apply (uc_result_call (K_of C) ap0 ap_cc ap_u _ k _ Hcall Hexn_ok
                 eq_refl eq_refl eq_refl eq_refl HKk).
        apply transport_fits. cbn. repeat constructor.
      * eapply (uc_result_call_wrap (K_of C) ap0 ap_cc
                  (with_continuation ap_u ffu) _ k _ ffu _ _ _
                  (call_rel_with _ _ _ k ffu Hcall) Hexn_ok eq_refl eq_refl
                  Hffue eq_refl eq_refl HKk).
        intros vs vs' Hvr Hck.
        destruct (vrel_one_plain vs vs' kr Hvr Hck) as (v & -> & ->).
        split; [ exact (proj2 (kinds_eqb_eq _ _) Hck) | ].
        intros wv.
        destruct tail;
        first
          [ exists ((ffu, v) :: wv), [Value_const (Poison kr)];
            split; [ reflexivity | split; [ reflexivity | ] ];
            apply vrel_dead; [ right; reflexivity | constructor ]
          | exists ((ffu, v) :: wv), [v];
            split; [ reflexivity | ];
            split; [ cbn; rewrite ?Pos.eqb_refl; reflexivity | ];
            apply vrel_dead; [ left; reflexivity | constructor ] ].
  - (* A result that the callee deletes. *)
    cbn [freshen_decisions wrapper_params flat_map app].
    destruct dk as [[kk' | ] | fk | kk'].
    + apply andb_true_iff in Hkp_ok as [Hmk Hkk]. subst mk.
      apply kind_eqb_eq in Hkk. subst kk'.
      cbn [is_kept forallb combine Bool.eqb andb kept_kinds flat_map
           wrapper_args wrapper_args_u app map length
           FlambdaFreshQuantifierSpec.names_from
           args_are_params].
      eapply (uc_result_call_wraps (K_of C) ap0 _ _ _ k
               (flat_map pshape_of [Pdesc (Decided (Keep kr)) true kr]) _ ff
               _ _ _ ffu _ _ _ (call_rel_with _ _ _ ff ffu Hcall) Hexn_ok
               eq_refl eq_refl Hffe Hffue eq_refl eq_refl HKk);
        [ reflexivity | ].
      intros vs vs' Hvr Hck.
      apply vrel_none in Hvr. subst vs'.
      apply kinds_eqb_eq in Hck. destruct vs; [ | discriminate Hck ].
      split; [ reflexivity | ].
      intros cv wv hv Hb. cbn in Hb. injection Hb as <-.
      exists wv. split; [ reflexivity | ].
      intros ws Hws0. cbn in Hws0. injection Hws0 as <-.
      exists [Value_const (Poison kr)]. split; [ reflexivity | ].
      apply vrel_plain. constructor.
    + apply negb_true_iff in Hkp_ok. subst mk.
      cbn [is_kept forallb combine Bool.eqb andb wrapper_args_u app map
           args_are_params].
      apply (uc_result_call (K_of C) ap0 ap_cc ap_u _ k _ Hcall Hexn_ok
               eq_refl eq_refl eq_refl eq_refl HKk).
      apply transport_refl.
    + discriminate Hret.
    + apply andb_true_iff in Hkp_ok as [Hmk Hkk]. subst mk.
      apply kind_eqb_eq in Hkk. subst kk'.
      cbn [is_kept forallb combine Bool.eqb andb kept_kinds flat_map
           wrapper_args wrapper_args_u app map length
           FlambdaFreshQuantifierSpec.names_from
           args_are_params].
      eapply (uc_result_call_wraps (K_of C) ap0 _ _ _ k
               (flat_map pshape_of [Pdesc (Keep_unused kr) true kr]) _ ff
               _ _ _ ffu _ _ _ (call_rel_with _ _ _ ff ffu Hcall) Hexn_ok
               eq_refl eq_refl Hffe Hffue eq_refl eq_refl HKk);
        [ reflexivity | ].
      intros vs vs' Hvr Hck.
      apply vrel_none in Hvr. subst vs'.
      apply kinds_eqb_eq in Hck. destruct vs; [ | discriminate Hck ].
      split; [ reflexivity | ].
      intros cv wv hv Hb. cbn in Hb. injection Hb as <-.
      exists wv. split; [ reflexivity | ].
      intros ws Hws0. cbn in Hws0. injection Hws0 as <-.
      exists [Value_const (Poison kr)]. split; [ reflexivity | ].
      apply vrel_dead; [ left; reflexivity | constructor ].
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
         (C : finite_map (list pdesc * bool)) (blocked : list continuation)
         (ctx : option code_id) (ce : finite_map (list node)) (is_exn : bool),
         cinv C blocked en enu ->
         calls_handler F sol cannot_change code_deps unit_code_ids codes
           unboxed (env_kinds en) C (handler_pdescs sol unboxed is_exn h) h
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
             ∧ rebuild_handler_u true tail F sol cannot_change code_deps pdc
                 unit_code_ids codes unboxed ffu [] enu dead_check
                 (handler_decisions true sol unboxed is_exn h) h
               = Handler (map fst ps') md' hu
             ∧ params_rel2 unboxed dead
                 (flat_map pshape_of (handler_pdescs sol unboxed is_exn h))
                 ps ps'
             ∧ hm_params_kinds md = map snd ps
             ∧ hm_params_kinds md' = map snd ps'
             ∧ map snd ps
               = flat_map kept_kind (handler_pdescs sol unboxed is_exn h)
             ∧ map pd_decision (handler_pdescs sol unboxed is_exn h)
               = handler_decisions true sol unboxed is_exn h
             ∧ map pd_keep (handler_pdescs sol unboxed is_exn h)
               = handler_mask sol is_exn h
             ∧ uc_result (K_of C) (free_continuations_of_handler h) hp hu).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en enu C blocked ctx ce Hci Hloc Hnames Hedges.
    cbn [calls_expr] in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hkind Hbind].
    destruct (names_let x kind d body Hnames) as (Hnb & [Hxf Hxd] & Hdn).
    destruct Hedges as [Hnamed_edges Hbody_edges].
    assert (Hci' : cinv C blocked (bind_variables en [(x, kind)])
                     (ubind_variables enu [(x, kind)]))
      by (apply cinv_bind; [ exact Hci | intros y k [Heq | []];
                             injection Heq as <- <-; exact Hkind ]).
    specialize (IH (bind_variables en [(x, kind)])
                  (ubind_variables enu [(x, kind)]) C blocked ctx ce Hci'
                  Hbody Hnb Hbody_edges).
    cbn [free_continuations_of_expr]. unfold cc_rebuild, u_rebuild in *.
    cbn [rebuild_expr_cc rebuild_expr_u].
    unfold is_pending_projection.
    rewrite (ci_proj _ _ _ _ Hci), (ci_proj_u _ _ _ _ Hci).
    rewrite (rebuild_named_kinds sol en (ue_env enu) x d
               (eq_sym (ci_kinds _ _ _ _ Hci))).
    assert (Hkok : ∀ y, kind_ok (kind_of_variable (ue_env enu) y) = true).
    { intros y. unfold kind_of_variable. rewrite (ci_kinds _ _ _ _ Hci).
      exact (ci_kinds_ok _ _ _ _ Hci y). }
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
          with (body := rebuild_expr_u true tail F sol cannot_change code_deps
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
          rewrite (ci_kinds _ _ _ _ Hci). exact Hkind'. }
        unfold rebuild_unboxed_binding. rewrite Hl.
        destruct IH as (IHuc & IHfc & IHbs & IHincl).
        destruct (lets_free_continuations (concat ls)
                    (rebuild_expr_u true tail F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body))
          as [Hfc Hbs].
        split.
        { refine (uc_alloc unboxed dead CSf _ x _ fields ls _ _ Hx Hxf _ _
                    IHuc).
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
          with (body := rebuild_expr_u true tail F sol cannot_change code_deps
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
                    (rebuild_expr_u true tail F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body))
          as [Hfc Hbs].
        split.
        { refine (uc_alloc unboxed dead CSf _ x _ fields _ _ _ Hx Hxf _ _
                    IHuc).
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
        rewrite (ci_kinds _ _ _ _ Hci), Hyused.
        unfold rebuild_unboxed_binding.
        destruct (bind_fields_pairs fields fields_y
                    (rebuild_expr_u true tail F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body)
                    Hflat)
          as (pairs & Hentries & Hl).
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hl.
        destruct (lets_free_continuations (pairs_lets pairs)
                    (rebuild_expr_u true tail F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body))
          as [Hfc Hbs].
        split; [ exact (uc_alias unboxed dead CSf _ x y fields fields_y pairs
                          _ _ Hx Hxf Hy Hne Hentries IHuc) | ].
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
                    (rebuild_expr_u true tail F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body)
                    Halike)
          as (pairs & Hentries & Hl).
        rewrite Hl.
        destruct (lets_free_continuations (pairs_lets pairs)
                    (rebuild_expr_u true tail F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu []
                       (ubind_variables enu [(x, Value)]) body))
          as [Hfc Hbs].
        split.
        -- apply (uc_load unboxed dead CSf _ x Value _
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
              apply (uc_read unboxed dead CSf _ x kind
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
    intros bindings body IHcode IH en enu C blocked ctx ce Hci Hloc
      [Hvars Hconts] Hedges.
    cbn [calls_expr] in Hloc. apply andb_true_iff in Hloc as [Hbs Hbody].
    destruct Hedges as [Hb_edges Hbody_edges].
    cbn [variables_of continuations_of] in Hvars, Hconts.
    destruct (IH (bind_variables en []) (ubind_variables enu []) C blocked
                ctx ce (cinv_bind C blocked en enu [] Hci
                          (fun y k (H : (y, k) ∈ []) => match H with end)))
      as (IHuc & IHfc & IHbs & IHincl).
    { exact Hbody. }
    { split; intros; [ apply Hvars | apply Hconts ]; apply in_or_app; right;
        assumption. }
    { exact Hbody_edges. }
    destruct (bindings_rebuilt en enu C blocked bindings IHcode Hci Hbs)
      as (Hub & Hbfc & Hbbs & Hbincl).
    { intros z Hz. apply Hvars, in_or_app. left. exact Hz. }
    { intros k Hk. apply Hconts, in_or_app. left. exact Hk. }
    { exact Hb_edges. }
    rewrite cc_rebuild_let_static, u_rebuild_let_static.
    split; [ apply uc_static; [ exact Hub | exact IHuc ] | ].
    cbn [free_continuations_of_expr binds_statics].
    split; [ f_equal; [ exact Hbfc | exact IHfc ] | ].
    split; [ f_equal; [ exact Hbbs | exact IHbs ] | ].
    intros k Hk.
    apply in_app_or in Hk as [Hk | Hk]; apply in_or_app;
      [ left; exact (Hbincl k Hk) | right; exact (IHincl k Hk) ].
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en enu C blocked ctx ce Hci Hloc
      [Hvars Hconts] Hedges.
    cbn [calls_expr] in Hloc. cbv zeta in Hloc.
    apply andb_true_iff in Hloc as [Hh Hbody].
    destruct Hedges as [Hh_edges Hbody_edges].
    cbn [variables_of continuations_of] in Hvars, Hconts.
    set (is_exn := handler_is_exn_handler h) in *.
    set (ps := handler_pdescs sol unboxed is_exn h) in *.
    destruct (IHh en enu C blocked ctx ce is_exn Hci) with (dead_check := true)
      as (ps_c & ps_u & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd'
          & Hkinds & Hdecs & Hmask & Hhres).
    { exact Hh. }
    { split; intros; [ apply Hvars | apply Hconts ];
        [ apply in_or_app; left | right; apply in_or_app; left ];
        assumption. }
    { exact Hh_edges. }
    fold ps in Hdecs, Hmask, Hkinds, Hrel.
    assert (Hps_ok : forallb pdesc_ok ps = true).
    { destruct h as [params md0 handler]. cbn [calls_handler] in Hh.
      apply andb_true_iff in Hh as [Hh _].
      apply andb_true_iff in Hh as [_ Hh]. exact Hh. }
    pose proof (cinv_conts C blocked en enu [(k, (ps, true))] Hci
                  (fun k0 ps0 h0 (H : (k0, (ps0, h0)) ∈ [(k, (ps, true))]) =>
                     match H with
                     | or_introl Heq =>
                       eq_ind_r (fun p => forallb pdesc_ok p = true) Hps_ok
                         (f_equal (fun t => fst (snd t)) (eq_sym Heq))
                     | or_intror H' => match H' with end
                     end))
      as Hci'.
    cbn [map app] in Hci'. rewrite Hdecs, Hmask in Hci'.
    destruct (IH _ _ ((k, (ps, true)) :: C) (without [k] blocked) ctx
                ((k, FlambdaReaperTraversal.handler_params h) :: ce) Hci')
      as (IHuc & IHfc & IHbs & IHincl).
    { exact Hbody. }
    { split; intros; [ apply Hvars | apply Hconts ];
        [ apply in_or_app; right | right; apply in_or_app; right ];
        assumption. }
    { exact Hbody_edges. }
    rewrite cc_rebuild_let_cont, u_rebuild_let_cont. fold is_exn. cbv zeta.
    assert (Hfc' :
      free_continuations_of_expr
        (u_rebuild
           (ubind_continuations enu
              [(k, handler_decisions true sol unboxed is_exn h)])
           body)
      = free_continuations_of_expr
          (cc_rebuild (bind_continuations en [(k, handler_mask sol is_exn h)])
             body))
      by (symmetry; exact IHfc).
    rewrite Hfc'.
    destruct (among k (free_continuations_of_expr
                         (cc_rebuild
                            (bind_continuations en
                               [(k, handler_mask sol is_exn h)])
                            body)))
      eqn:Hk.
    + rewrite Hp, Hu. destruct Hhres as (Hhuc & Hhfc & Hhbs & Hhincl).
      split.
      { apply (uc_let_cont unboxed dead CSf (K_of C) k (flat_map pshape_of ps)
                 ps_c ps_u md md' hp hu); try assumption.
        cbn [K_of map] in IHuc. rewrite Hkinds. exact IHuc. }
      cbn [free_continuations_of_expr free_continuations_of_handler
           binds_statics handler_binds_statics].
      split; [ f_equal; [ exact Hhfc | f_equal; exact IHfc ] | ].
      split; [ f_equal; [ exact Hhbs | exact IHbs ] | ].
      intros k' Hk'.
      apply in_app_or in Hk' as [Hk' | Hk']; apply in_or_app;
        [ left; exact (Hhincl k' Hk') | right ].
      apply in_without_iff in Hk' as [Hk' Hne].
      apply in_without_iff. split; [ exact (IHincl k' Hk') | exact Hne ].
    + split; [ | split; [ exact IHfc | split; [ exact IHbs | ] ] ].
      * apply (uc_agree unboxed dead CSf _ _ _ IHuc). intros k' Hk'.
        cbn [K_of map]. rewrite lookup_cons.
        destruct (Pos.eqb k' k) eqn:E; [ | reflexivity ].
        apply Pos.eqb_eq in E. subst k'. exfalso.
        assert (Ht : among k
                       (free_continuations_of_expr
                          (cc_rebuild
                             (bind_continuations en
                                [(k, handler_mask sol is_exn h)])
                             body))
                     = true) by (apply among_In; exact Hk').
        rewrite Ht in Hk. discriminate.
      * intros k' Hk'. cbn [free_continuations_of_expr]. apply in_or_app.
        right. apply in_without_iff. split; [ exact (IHincl k' Hk') | ].
        intros [Heq | []]. subst k'.
        assert (Ht : among k
                       (free_continuations_of_expr
                          (cc_rebuild
                             (bind_continuations en
                                [(k, handler_mask sol is_exn h)])
                             body))
                     = true) by (apply among_In; exact Hk').
        rewrite Ht in Hk. discriminate.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en enu C blocked ctx ce Hci Hloc
      [Hvars Hconts] Hedges.
    cbn [calls_expr] in Hloc. cbv zeta in Hloc.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite forallb_forall in Hhs.
    destruct Hedges as [Hhs_edges Hbody_edges].
    cbn [variables_of continuations_of] in Hvars, Hconts.
    set (D := map (fun '(k, h) =>
                     (k, (handler_pdescs sol unboxed false h, true)))
                handlers) in *.
    assert (Hhs' : ∀ k h, (k, h) ∈ handlers ->
              calls_handler F sol cannot_change code_deps unit_code_ids codes
                unboxed (env_kinds en) (D ++ C)
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
      cbn [calls_handler] in Hhs'.
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
      cbn [calls_handler] in Hhs'.
      apply andb_true_iff in Hhs' as [Hhs' _].
      apply andb_true_iff in Hhs' as [_ Hok]. exact Hok. }
    pose proof (cinv_conts C blocked en enu D Hci HD) as Hci'.
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
    set (blocked' := without (map fst handlers) blocked).
    assert (Hci'' : cinv (D ++ C) blocked' en' enu').
    { unfold en', enu', blocked'. rewrite <- Hmk, <- Hdk, <- Hfst.
      exact Hci'. }
    clear Hci'. rename Hci'' into Hci'.
    destruct (IH en' enu' (D ++ C) blocked' ctx
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
                 (k, rebuild_handler_u true tail F sol cannot_change code_deps
                       pdc unit_code_ids codes unboxed ffu [] enu' false
                       (handler_decisions true sol unboxed false h) h))
            hs
        in
        uc_handlers unboxed dead CSf (K_of (D ++ C)) hps hus
          (map (fun '(k, h) =>
                  (k, Kshape
                        (flat_map pshape_of
                           (handler_pdescs sol unboxed false h))
                        (flat_map kept_kind
                           (handler_pdescs sol unboxed false h))
                        true))
             hs)
        ∧ handlers_bind_statics hps = handlers_bind_statics hus
        ∧ flat_map (fun '(_, h) => free_continuations_of_handler h) hps
          = flat_map (fun '(_, h) => free_continuations_of_handler h) hus
        ∧ incl (flat_map (fun '(_, h) => free_continuations_of_handler h)
                  hps)
            (flat_map (fun '(_, h) => free_continuations_of_handler h) hs)
        ∧ map fst hps = map fst hus
        ∧ map fst hps = map fst hs).
    { induction hs as [ | [k h] hs IHhs]; intros Hsub; cbv zeta.
      - cbn [map flat_map]. split; [ apply uc_handlers_nil | ].
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ intros ? [] | split; reflexivity ].
      - destruct IHhs as (Hub_hs & Hbs_hs & Hfc_hs & Hincl_hs & Hfst_hs
                          & Hfst_hs').
        { intros k' h' Hin. apply Hsub. right. exact Hin. }
        assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
        destruct (IHh k h Hin en' enu' (D ++ C) blocked' ctx
                    (FlambdaReaperFlowProofs.rec_conts ce handlers) false
                    Hci')
          with (dead_check := false)
          as (ps_c & ps_u & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd'
              & Hkinds & _ & _ & Hhres).
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
        split; [ f_equal; [ exact Hhbs | exact Hbs_hs ] | ].
        split; [ f_equal; [ exact Hhfc | exact Hfc_hs ] | ].
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
    split; [ apply (uc_let_cont_rec unboxed dead CSf (K_of C) S);
             assumption | ].
    cbn [free_continuations_of_expr binds_statics].
    unfold handlers_bind_statics in Hbs_hs.
    split; [ f_equal; [ exact Hfst_hs | f_equal; [ exact Hfc_hs
                                                  | exact IHfc ] ] | ].
    split; [ f_equal; [ exact Hbs_hs | exact IHbs ] | ].
    intros k Hk. apply in_without_iff in Hk as [Hk Hne].
    apply in_without_iff. split.
    + apply in_app_or in Hk as [Hk | Hk]; apply in_or_app;
        [ left; exact (Hincl_hs k Hk) | right; exact (IHincl k Hk) ].
    + intros H. apply Hne. exact (eq_ind_r (fun l => k ∈ l) H Hfst_hs').
  - (* Apply *)
    intros ap en enu C blocked ctx ce Hci Hloc [Hvars Hconts] _.
    cbn [calls_expr] in Hloc. cbn [variables_of continuations_of] in Hvars,
      Hconts.
    apply (call_rebuilt en enu C blocked ap Hci); try assumption.
    + intros z Hz. exact (proj1 (Hvars z Hz)).
    + intros Heq. apply (proj1 (Hconts _ (or_intror (or_introl eq_refl)))).
      exact Heq.
    + intros Heq. apply (proj2 (Hconts _ (or_intror (or_introl eq_refl)))).
      exact Heq.
  - (* Apply_cont *)
    intros ac en enu C blocked ctx ce Hci Hloc [Hvars _] _.
    unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
    cbn [calls_expr] in Hloc. cbn [variables_of] in Hvars.
    unfold rebuild_apply_cont, rebuild_apply_cont_u.
    assert (Hnb : among (ac_continuation ac) blocked = false)
      by exact (not_blocked_ac C blocked en enu ac Hci Hloc).
    destruct (rewrite_apply_cont_rel2 en enu C blocked ac Hci Hnb
                (fun z Hz => proj1 (Hvars z Hz)) Hloc)
      as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
      rewrite Hp, Hu; [ exact (uc_result_invalid _ _) | ].
    split; [ constructor; exact Hrel | ].
    destruct Hrel as (Hk & Htrap & _).
    cbn [free_continuations_of_expr binds_statics].
    unfold apply_cont_continuations. rewrite Hk, Htrap.
    split; [ reflexivity | split; [ reflexivity | ] ].
    unfold rewrite_apply_cont in Hp.
    destruct (existsb _ _); [ discriminate Hp | ]. injection Hp as <-.
    intros k Hk'. exact Hk'.
  - (* Switch *)
    intros scrutinee arms en enu C blocked ctx ce Hci Hloc [Hvars _] _.
    cbn [calls_expr] in Hloc. apply andb_true_iff in Hloc as [Hs Harms].
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
              ∧ among (ac_continuation ac) blocked = false
              ∧ (∀ z, z ∈ variables_in_apply_cont ac -> fresh z)
              ∧ incl (apply_cont_continuations ac)
                  (free_continuations_of_expr (Switch scrutinee arms))).
    { intros d ac Hin. split; [ | split; [ | split ] ].
      - rewrite forallb_forall in Harms. exact (Harms _ Hin).
      - apply (not_blocked_ac C blocked en enu ac Hci).
        rewrite forallb_forall in Harms. exact (Harms _ Hin).
      - intros z Hz. apply Hfr0. apply in_or_app. right.
        apply in_flat_map. exists (d, ac). split; [ exact Hin | exact Hz ].
      - intros k Hk. cbn [free_continuations_of_expr]. apply in_flat_map.
        exists (d, ac). split; [ exact Hin | exact Hk ]. }
    assert (Harms' :
      ∀ l : list (Z * apply_cont),
        (∀ (d : Z) (ac : apply_cont), (d, ac) ∈ l -> (d, ac) ∈ arms) ->
        Forall2
          (fun arm arm' : Z * apply_cont =>
             fst arm' = fst arm
             ∧ ac_rel2 unboxed dead (K_of C) (snd arm) (snd arm')
             ∧ incl (apply_cont_continuations (snd arm))
                 (free_continuations_of_expr (Switch scrutinee arms)))
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
        as (Hok & Hnb & Hfr & Hinc).
      assert (Hrest := IHl (fun d' ac' Hin => Hl d' ac' (or_intror Hin))).
      destruct (rewrite_apply_cont_rel2 en enu C blocked ac Hci Hnb Hfr Hok)
        as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
        rewrite Hp, Hu; [ exact Hrest | ].
      constructor; [ | exact Hrest ].
      split; [ reflexivity | split; [ exact Hrel | ] ].
      unfold rewrite_apply_cont in Hp.
      destruct (existsb _ _); [ discriminate Hp | ]. injection Hp as <-.
      exact Hinc. }
    pose proof (Harms' _ Hcompact) as Hrel.
    rewrite (rewrite_simple_kinds sol_u (ue_env enu) en scrutinee
               (ci_kinds _ _ _ _ Hci)).
    assert (Hs' : plain_simple unboxed scrutinee = true).
    { apply (strong_simple2 scrutinee Hs). intros z Hz.
      apply Hfr0. apply in_or_app. left. exact Hz. }
    rewrite (rewrite_simple_u sol unboxed en scrutinee Hs').
    destruct Hrel as [ | arm arm' l l' Harm Hrel ];
      [ exact (uc_result_invalid _ _) | ].
    cbn [Keyed.is_empty].
    split.
    { apply uc_switch; [ apply rewrite_plain2;
                         [ exact (ci_kinds_ok _ _ _ _ Hci) | exact Hs' ] | ].
      constructor; [ destruct Harm as (Ha & Hb & _); split; assumption | ].
      clear -Hrel. induction Hrel as [ | a a' l l' (Ha & Hb & _) _ IH ];
        constructor; [ split; assumption | exact IH ]. }
    split; [ | split; [ reflexivity | ] ].
    + cbn [free_continuations_of_expr flat_map].
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
    intros en enu C blocked ctx ce _ _ _ _. exact (uc_result_invalid _ _).
  - (* Handler *)
    intros params metadata handler IH en enu C blocked ctx ce is_exn Hci
      Hloc [Hvars Hconts] Hedges dead_check.
    cbn [calls_handler] in Hloc.
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
    assert (Hci' : cinv C blocked (bind_variables en bound)
                     (ubind_variables enu bound))
      by (apply cinv_bind; [ exact Hci | intros y k Hin;
                             exact (proj2 (Hbk y k Hin)) ]).
    destruct (IH (bind_variables en bound) (ubind_variables enu bound) C
                blocked ctx ce Hci')
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

(* The statement of the fix and of its variant. *)
Lemma rebuild_unbox_modes_refines_cc (tail : bool)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_modes true tail (fun _ => false)
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox [] pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hcalls Hcheck.
  set (F := traverse_unit flag u e) in *.
  unfold calls_unboxing in Hcalls.
  apply andb_true_iff in Hcalls as [Hcalls Huses].
  pose proof (unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [Hdistinct Hfresh].
  pose proof (unboxed_fields_of_names_fresh F T (bound_code_my_closures e)
                dominated to_unbox
                (FlambdaNameAbstractionSpec.fresh_name (names_of e)))
    as Hnext.
  cbv zeta in Hnext. destruct Hnext as (Hnext & _ & _).
  unfold rebuild_unit_unbox_repr_modes, rebuild_unit_cc_from_tables,
    rebuild_unit_cc.
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
  destruct (proj1 (rebuild_uc tail F (solution_of_tables T)
                     (fun n => mem_node n cannot_change)
                     (unit_code_deps flag u e) pdc
                     (static_names code_ids_bound e) (bound_code_metadata e)
                     unboxed
                     (FlambdaNameAbstractionSpec.fresh_name (names_of e))
                     next dead noted deps cds (unit_kept e) (unit_codes e)
                     (static_names symbols_of_binding e) is_local Hcl Hslots
                     Hfields_read Hreads_found Hdead)
              e empty_env empty_unbox_env [] [] None (unit_conts u))
    as (Huc & _).
  - split.
    + intros k ps h Hl. discriminate Hl.
    + reflexivity.
    + intros x. reflexivity.
    + reflexivity.
    + reflexivity.
    + reflexivity.
    + intros k _. reflexivity.
    + intros k _. left. reflexivity.
    + intros k Hk. discriminate Hk.
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
  - exact (uc_refines unboxed Hdistinct dead _ _ _ Huc).
Qed.

Theorem rebuild_unbox_fixed_refines_cc :
  rebuild_unbox_fixed_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc.
  exact (rebuild_unbox_modes_refines_cc false flag is_local u e T
           cannot_change dominated to_unbox pdc).
Qed.

Theorem rebuild_unbox_fixed_tail_refines_cc :
  rebuild_unbox_fixed_tail_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc.
  exact (rebuild_unbox_modes_refines_cc true flag is_local u e T
           cannot_change dominated to_unbox pdc).
Qed.

Theorem rebuild_unbox_fixed_refines_observably :
  rebuild_unbox_fixed_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc Hslots
    Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs.rebuild_cc_refines_observably
              flag is_local u e T cannot_change pdc Hslots Hproj Hfrag Hany
              Hrk Hwf Hone Hclosed Hexact Hccc Hwid)
           (rebuild_unbox_fixed_refines_cc flag is_local u e T cannot_change
              dominated to_unbox pdc Hslots Hfrag Hclosed Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_fixed_tail_refines_observably :
  rebuild_unbox_fixed_tail_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc Hslots
    Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs.rebuild_cc_refines_observably
              flag is_local u e T cannot_change pdc Hslots Hproj Hfrag Hany
              Hrk Hwf Hone Hclosed Hexact Hccc Hwid)
           (rebuild_unbox_fixed_tail_refines_cc flag is_local u e T
              cannot_change dominated to_unbox pdc Hslots Hfrag Hclosed Hcalls
              Hcheck)).
Qed.

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
    = TailExample.rebuilt_fixed)
    by (vm_compute; reflexivity).
  assert (Htail :
    rebuild_unit_unbox_fixed_tail TailExample.facts TailExample.tables
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
  split; [ exact Hfixed | split; [ exact Htail | ] ].
  split.
  - rewrite <- Hfixed.
    apply (rebuild_unbox_fixed_refines_observably Auto Examples.no_local
             Examples.interface TailExample.unit_expr TailExample.tables
             TailExample.cannot_change [] [] true); try assumption.
    left. intros f. reflexivity.
  - rewrite <- Htail.
    apply (rebuild_unbox_fixed_tail_refines_observably Auto
             Examples.no_local Examples.interface TailExample.unit_expr
             TailExample.tables TailExample.cannot_change [] [] true);
      try assumption.
    left. intros f. reflexivity.
Qed.
