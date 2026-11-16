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

(* Proofs of the statements of FlambdaAlphaGroupSpec.v. Nothing here
   needs review.

   The rule read both ways. Exchanging the continuations of a group for
   fresh ones is a renaming that fixes the free names of the whole
   term, so each term is alpha-equivalent to the one whose group binds
   the fresh names ([group_opened]). Two such terms bind the same
   continuations. From left to right the comparison accepts them, so
   it accepts the handlers one by one and the bodies with no names
   paired ([group_eqb_alpha]), and what it accepts is alpha-equivalent
   (FlambdaCompareConverseProofs.v, FlambdaAlphaFreshProofs.v). From
   right to left the body is replaced under the group, and then the
   handlers one after another ([alpha_handlers_group]).

   The abstraction. A handler is equal to another when it has the same
   metadata and is equal as an abstraction of its parameters; that
   relation is an equivalence by the laws of abstractions
   (FlambdaNameAbstractionProofs.abstraction_nominal), and the other
   laws, which are about continuations, come from those of closed
   expressions, the two sorts of renaming being independent
   ([rename_sorts_commute]). The theorem is proved as those of
   FlambdaNominalInstancesProofs.v are: each side is opened by its
   permutation, which is a renaming of the whole term that fixes its
   free names, and the two opened terms bind the same continuations
   and have equal scopes. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaContextSpec FlambdaBindersSpec
  FlambdaBindersProofs FlambdaFreeNamesSpec FlambdaFreeNamesProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaCompareSpec FlambdaCompareProofs
  FlambdaCompareAlphaSpec FlambdaCompareAlphaProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaNominalInstancesSpec FlambdaNominalInstancesProofs
  FlambdaFreshQuantifierSpec FlambdaFreshQuantifierProofs
  FlambdaFreshBindersSpec FlambdaFreshBindersProofs
  FlambdaCompareConverseSpec FlambdaCompareConverseProofs
  FlambdaAlphaFreshSpec FlambdaAlphaFreshProofs FlambdaAlphaGroupSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* The handlers of a group, one after another                               *)
(* ======================================================================== *)

Lemma alpha_handlers_bound_recursive (h h' : continuation_handler) :
  handler_code_closed h = true ->
  handler_code_closed h' = true ->
  handler_binders_distinct h = true ->
  handler_binders_distinct h' = true ->
  alpha_handlers h h' ->
  ∀ (before : list (continuation * continuation_handler)) (k : continuation)
    (after : list (continuation * continuation_handler)) (body : expr),
    alpha_equivalent (Let_cont_recursive (before ++ (k, h) :: after) body)
      (Let_cont_recursive (before ++ (k, h') :: after) body).
Proof.
  destruct h as [params metadata handler],
    h' as [params' metadata' handler'].
  cbn [handler_code_closed handler_binders_distinct alpha_handlers].
  intros Hclosed Hclosed' Hdistinct Hdistinct' (Hlength & <- & Hfresh) before
    k after body.
  apply andb_true_iff in Hdistinct as [Hparams _].
  apply andb_true_iff in Hdistinct' as [Hparams' _].
  destruct
    (for_fresh_names_elim _ _
       ((params ++ free_variables_of_expr handler)
        ++ params' ++ free_variables_of_expr handler')
       Hfresh)
    as (zs & Hcount & Houtside & Hscopes).
  apply distinct_outside_app in Houtside as [Houtside Houtside'].
  exact
    (alpha_recursive_handler_at before k params params' zs metadata handler
       handler' after body Hclosed Hclosed' Hparams Hparams' Hcount
       (eq_trans Hcount Hlength) Houtside Houtside' Hscopes).
Qed.

(* [done] is what has been replaced. The two lists have the same
   continuations. *)
Lemma alpha_handlers_group (body : expr)
    (hs : list (continuation * continuation_handler)) :
  ∀ hs' done : list (continuation * continuation_handler),
    map fst hs = map fst hs' ->
    Forall2 alpha_handlers (map snd hs) (map snd hs') ->
    forallb (fun '(_, h) => handler_code_closed h) hs = true ->
    forallb (fun '(_, h) => handler_code_closed h) hs' = true ->
    forallb (fun '(_, h) => handler_binders_distinct h) hs = true ->
    forallb (fun '(_, h) => handler_binders_distinct h) hs' = true ->
    alpha_equivalent (Let_cont_recursive (done ++ hs) body)
      (Let_cont_recursive (done ++ hs') body).
Proof.
  induction hs as [ | [k h] hs IH];
    intros [ | [k' h'] hs'] done Hnames Hall Hclosed Hclosed' Hdistinct
      Hdistinct';
    cbn [map fst snd] in Hnames, Hall; try discriminate Hnames;
    [ apply Alpha_refl | ].
  injection Hnames as <- Hnames.
  inversion Hall as [ | first first' rest rest' Hfirst Hrest]; subst.
  cbn [forallb] in Hclosed, Hclosed', Hdistinct, Hdistinct'.
  apply andb_true_iff in Hclosed as [Hclosed_first Hclosed].
  apply andb_true_iff in Hclosed' as [Hclosed_first' Hclosed'].
  apply andb_true_iff in Hdistinct as [Hdistinct_first Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hdistinct_first' Hdistinct'].
  apply Alpha_trans with (Let_cont_recursive (done ++ (k, h') :: hs) body).
  - apply alpha_handlers_bound_recursive; assumption.
  - replace (done ++ (k, h') :: hs) with ((done ++ [(k, h')]) ++ hs)
      by (rewrite <- app_assoc; reflexivity).
    replace (done ++ (k, h') :: hs') with ((done ++ [(k, h')]) ++ hs')
      by (rewrite <- app_assoc; reflexivity).
    apply IH; assumption.
Qed.

Lemma handlers_eqb_alpha_handlers
    (hs : list (continuation * continuation_handler)) :
  ∀ hs' : list (continuation * continuation_handler),
    forallb (fun '(_, h) => handler_code_closed h) hs = true ->
    forallb (fun '(_, h) => handler_code_closed h) hs' = true ->
    forallb (fun '(_, h) => handler_binders_distinct h) hs = true ->
    forallb (fun '(_, h) => handler_binders_distinct h) hs' = true ->
    forall2b (fun '(_, h) '(_, h') => handler_eqb [] [] h h') hs hs' = true ->
    Forall2 alpha_handlers (map snd hs) (map snd hs').
Proof.
  induction hs as [ | [k h] hs IH];
    intros [ | [k' h'] hs'] Hclosed Hclosed' Hdistinct Hdistinct' Heq;
    cbn [forall2b] in Heq; try discriminate Heq; [ constructor | ].
  cbv beta iota in Heq. apply andb_true_iff in Heq as [Hfirst Hrest].
  cbn [forallb] in Hclosed, Hclosed', Hdistinct, Hdistinct'.
  apply andb_true_iff in Hclosed as [Hclosed_first Hclosed].
  apply andb_true_iff in Hclosed' as [Hclosed_first' Hclosed'].
  apply andb_true_iff in Hdistinct as [Hdistinct_first Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hdistinct_first' Hdistinct'].
  cbn [map snd]. constructor;
    [ apply handler_eqb_alpha_handlers; assumption | apply IH; assumption ].
Qed.

(* ======================================================================== *)
(* Two groups that bind the same continuations                              *)
(* ======================================================================== *)

Lemma group_eqb_alpha (zks : list continuation)
    (hs hs' : list (continuation * continuation_handler))
    (body body' : expr) :
  code_closed (Let_cont_recursive hs body) = true ->
  code_closed (Let_cont_recursive hs' body') = true ->
  binders_distinct (Let_cont_recursive hs body) = true ->
  binders_distinct (Let_cont_recursive hs' body') = true ->
  map fst hs = zks ->
  map fst hs' = zks ->
  expr_eqb [] [] (Let_cont_recursive hs body) (Let_cont_recursive hs' body')
  = true ->
  Forall2 alpha_handlers (map snd hs) (map snd hs')
  ∧ alpha_equivalent body body'.
Proof.
  intros Hclosed Hclosed' Hdistinct Hdistinct' Hnames Hnames' Heq.
  cbn [expr_eqb] in Heq. rewrite Hnames, Hnames' in Heq.
  unfold combine_names in Heq. rewrite Nat.eqb_refl in Heq.
  apply andb_true_iff in Heq as [Hgroup Hbodies].
  apply (expr_eqb_diagonal [] (combine zks zks ++ []) _ _ diagonal_nil
           (combine_diagonal zks [] diagonal_nil))
    in Hbodies.
  apply
    (forall2b_impl _ (fun '(_, h) '(_, h') => handler_eqb [] [] h h'))
    in Hgroup;
    [ | intros [k h] [k' h'];
        exact (handler_eqb_diagonal [] (combine zks zks ++ []) h h'
                 diagonal_nil (combine_diagonal zks [] diagonal_nil)) ].
  cbn [code_closed] in Hclosed, Hclosed'.
  apply andb_true_iff in Hclosed as [Hclosed_handlers Hclosed_body].
  apply andb_true_iff in Hclosed' as [Hclosed_handlers' Hclosed_body'].
  cbn [binders_distinct] in Hdistinct, Hdistinct'.
  apply andb_true_iff in Hdistinct as [Hdistinct Hdistinct_body].
  apply andb_true_iff in Hdistinct as [_ Hdistinct_handlers].
  apply andb_true_iff in Hdistinct' as [Hdistinct' Hdistinct_body'].
  apply andb_true_iff in Hdistinct' as [_ Hdistinct_handlers'].
  split;
    [ apply handlers_eqb_alpha_handlers
    | apply expr_eqb_alpha_equivalent ];
    assumption.
Qed.

(* A group with its continuations exchanged for new ones. *)
Lemma group_opened (zks : list continuation)
    (hs : list (continuation * continuation_handler)) (body : expr) :
  code_closed (Let_cont_recursive hs body) = true ->
  binders_distinct (Let_cont_recursive hs body) = true ->
  opens_at (map fst hs) zks
    (free_continuations_of_expr (Let_cont_recursive hs body)) ->
  alpha_equivalent (Let_cont_recursive hs body)
    (Let_cont_recursive
       (rename_handlers (swap_continuations (map fst hs) zks) hs)
       (rename_expr (swap_continuations (map fst hs) zks) body))
  ∧ code_closed
      (Let_cont_recursive
         (rename_handlers (swap_continuations (map fst hs) zks) hs)
         (rename_expr (swap_continuations (map fst hs) zks) body))
    = true
  ∧ binders_distinct
      (Let_cont_recursive
         (rename_handlers (swap_continuations (map fst hs) zks) hs)
         (rename_expr (swap_continuations (map fst hs) zks) body))
    = true
  ∧ map fst (rename_handlers (swap_continuations (map fst hs) zks) hs) = zks.
Proof.
  intros Hclosed Hdistinct Hopens.
  set (s := swap_continuations (map fst hs) zks).
  destruct (group_fixes zks hs body Hclosed Hopens)
    as [Hvariables Hcontinuations].
  split;
    [ exact (Alpha_rename_by_relation s _ Hclosed Hvariables Hcontinuations)
    | ].
  split;
    [ exact (eq_trans
               (code_closed_equivariant s (Let_cont_recursive hs body))
               Hclosed)
    | ].
  split;
    [ exact (eq_trans
               (binders_distinct_equivariant s (Let_cont_recursive hs body))
               Hdistinct)
    | ].
  rewrite map_fst_rename_handlers. exact (opens_at_moved _ _ _ Hopens).
Qed.

(* The continuations of a group whose binders are distinct. *)
Lemma group_names_distinct
    (hs : list (continuation * continuation_handler)) (body : expr) :
  binders_distinct (Let_cont_recursive hs body) = true ->
  distinct (map fst hs) = true.
Proof.
  cbn [binders_distinct]. intros Hdistinct.
  apply andb_true_iff in Hdistinct as [Hdistinct _].
  apply andb_true_iff in Hdistinct as [Hdistinct _].
  apply andb_true_iff in Hdistinct as [_ Hdistinct]. exact Hdistinct.
Qed.

Theorem alpha_let_cont_recursive_iff
  : alpha_let_cont_recursive_iff_statement.
Proof.
  intros hs hs' body body' Hclosed Hclosed' Hdistinct Hdistinct'.
  pose proof (group_names_distinct hs body Hdistinct) as Hnames.
  pose proof (group_names_distinct hs' body' Hdistinct') as Hnames'.
  (* What new names give for both groups. *)
  assert (Hopened :
    ∀ zks : list continuation,
      length hs = length hs' ->
      length zks = length hs ->
      distinct_outside
        ((map fst hs
          ++ free_continuations_of_expr (Let_cont_recursive hs body))
         ++ map fst hs'
         ++ free_continuations_of_expr (Let_cont_recursive hs' body'))
        zks ->
      opens_at (map fst hs) zks
        (free_continuations_of_expr (Let_cont_recursive hs body))
      ∧ opens_at (map fst hs') zks
          (free_continuations_of_expr (Let_cont_recursive hs' body'))).
  { intros zks Hlength Hcount Houtside.
    apply distinct_outside_app in Houtside as [Houtside Houtside'].
    split; (split; [ assumption | ]).
    - split; [ rewrite length_map; exact Hcount | exact Houtside ].
    - split;
        [ rewrite length_map; exact (eq_trans Hcount Hlength)
        | exact Houtside' ]. }
  split.
  - intros Halpha.
    assert (Hlength : length hs = length hs').
    { pose proof (alpha_equivalent_accepted _ _ Halpha) as Heq.
      cbn [expr_eqb] in Heq.
      destruct (combine_names (map fst hs) (map fst hs') [])
        as [km | ] eqn:Hcombine; [ | discriminate Heq ].
      pose proof (combine_names_length _ _ _ _ Hcombine) as Hsame.
      rewrite !length_map in Hsame. exact Hsame. }
    split; [ exact Hlength | ].
    exists
      ((map fst hs
        ++ free_continuations_of_expr (Let_cont_recursive hs body))
       ++ map fst hs'
       ++ free_continuations_of_expr (Let_cont_recursive hs' body')).
    intros zks Hcount Houtside.
    destruct (Hopened zks Hlength Hcount Houtside) as [Hopens Hopens'].
    destruct (group_opened zks hs body Hclosed Hdistinct Hopens)
      as (Hrenamed & Hclosed1 & Hdistinct1 & Hmoved).
    destruct (group_opened zks hs' body' Hclosed' Hdistinct' Hopens')
      as (Hrenamed' & Hclosed1' & Hdistinct1' & Hmoved').
    apply (group_eqb_alpha zks); try assumption.
    apply alpha_equivalent_accepted.
    apply Alpha_trans with (Let_cont_recursive hs body);
      [ apply Alpha_sym; exact Hrenamed | ].
    apply Alpha_trans with (Let_cont_recursive hs' body');
      [ exact Halpha | exact Hrenamed' ].
  - intros [Hlength Hfresh].
    destruct
      (for_fresh_names_elim _ _
         ((map fst hs
           ++ free_continuations_of_expr (Let_cont_recursive hs body))
          ++ map fst hs'
          ++ free_continuations_of_expr (Let_cont_recursive hs' body'))
         Hfresh)
      as (zks & Hcount & Houtside & Hhandlers & Hbodies).
    destruct (Hopened zks Hlength Hcount Houtside) as [Hopens Hopens'].
    destruct (group_opened zks hs body Hclosed Hdistinct Hopens)
      as (Hrenamed & Hclosed1 & Hdistinct1 & Hmoved).
    destruct (group_opened zks hs' body' Hclosed' Hdistinct' Hopens')
      as (Hrenamed' & Hclosed1' & Hdistinct1' & Hmoved').
    set (s := swap_continuations (map fst hs) zks) in *.
    set (s' := swap_continuations (map fst hs') zks) in *.
    apply Alpha_trans
      with (Let_cont_recursive (rename_handlers s hs) (rename_expr s body));
      [ exact Hrenamed | ].
    apply Alpha_trans
      with (Let_cont_recursive (rename_handlers s' hs')
              (rename_expr s' body'));
      [ | apply Alpha_sym; exact Hrenamed' ].
    cbn [code_closed] in Hclosed1, Hclosed1'.
    apply andb_true_iff in Hclosed1 as [Hclosed_handlers _].
    apply andb_true_iff in Hclosed1' as [Hclosed_handlers' _].
    cbn [binders_distinct] in Hdistinct1, Hdistinct1'.
    apply andb_true_iff in Hdistinct1 as [Hdistinct1 _].
    apply andb_true_iff in Hdistinct1 as [_ Hdistinct_handlers].
    apply andb_true_iff in Hdistinct1' as [Hdistinct1' _].
    apply andb_true_iff in Hdistinct1' as [_ Hdistinct_handlers'].
    apply Alpha_trans
      with (Let_cont_recursive (rename_handlers s hs) (rename_expr s' body')).
    + exact
        (Alpha_context (Let_cont_recursive_body (rename_handlers s hs) Hole)
           _ _ Hbodies).
    + exact
        (alpha_handlers_group (rename_expr s' body') (rename_handlers s hs)
           (rename_handlers s' hs') [] (eq_trans Hmoved (eq_sym Hmoved'))
           Hhandlers Hclosed_handlers Hclosed_handlers' Hdistinct_handlers
           Hdistinct_handlers').
Qed.

(* ======================================================================== *)
(* Lists of things that contain names                                       *)
(* ======================================================================== *)

Section Lists.

Context {T : Type}.

Variable N : contains_names T.

Hypothesis HN : nominal N.

Lemma list_equal_refl (ts : list T) : Forall2 (equal N) ts ts.
Proof.
  induction ts as [ | t ts IH]; constructor;
    [ apply (equal_refl N HN) | exact IH ].
Qed.

Lemma list_equal_sym (ts ts' : list T) :
  Forall2 (equal N) ts ts' -> Forall2 (equal N) ts' ts.
Proof.
  induction 1 as [ | t t' ts ts' Ht _ IH]; constructor;
    [ apply (equal_sym N HN), Ht | exact IH ].
Qed.

Lemma list_equal_trans (ts ts' ts'' : list T) :
  Forall2 (equal N) ts ts' -> Forall2 (equal N) ts' ts'' ->
  Forall2 (equal N) ts ts''.
Proof.
  intros Hall. revert ts''.
  induction Hall as [ | t t' ts ts' Ht _ IH]; intros ts'' Hall';
    inversion Hall' as [ | first first' rest rest' Ht' Hrest]; subst;
    [ constructor | ].
  constructor;
    [ exact (equal_trans N HN _ _ _ Ht Ht') | apply IH, Hrest ].
Qed.

Lemma list_equal_equivariant (p : permutation) (ts ts' : list T) :
  Forall2 (equal N) ts ts' ->
  Forall2 (equal N) (map (apply_renaming N p) ts)
    (map (apply_renaming N p) ts').
Proof.
  induction 1 as [ | t t' ts ts' Ht _ IH]; cbn [map]; constructor;
    [ apply (equal_equivariant N HN), Ht | exact IH ].
Qed.

Lemma list_free_names_equivariant (p : permutation) (ts : list T)
    (name : positive) :
  among (permute p name)
    (flat_map (free_names N) (map (apply_renaming N p) ts))
  = among name (flat_map (free_names N) ts).
Proof.
  induction ts as [ | t ts IH]; [ reflexivity | ]. cbn [map flat_map].
  rewrite !among_app, IH, (free_names_equivariant N HN). reflexivity.
Qed.

Lemma list_free_names_equal (ts ts' : list T) :
  Forall2 (equal N) ts ts' ->
  same_names (flat_map (free_names N) ts) (flat_map (free_names N) ts').
Proof.
  induction 1 as [ | t t' ts ts' Ht _ IH]; intros name; [ reflexivity | ].
  cbn [flat_map].
  rewrite !among_app, (IH name), (free_names_equal N HN t t' Ht name).
  reflexivity.
Qed.

Lemma list_apply_renaming_fixes (p : permutation) (ts : list T) :
  fixes p (flat_map (free_names N) ts) ->
  Forall2 (equal N) (map (apply_renaming N p) ts) ts.
Proof.
  induction ts as [ | t ts IH]; intros Hfixes; [ constructor | ].
  cbn [map]. constructor.
  - apply (apply_renaming_fixes N HN). intros name Hname. apply Hfixes.
    cbn [flat_map]. rewrite among_app, Hname. reflexivity.
  - apply IH. intros name Hname. apply Hfixes.
    cbn [flat_map]. rewrite among_app, Hname. apply orb_true_r.
Qed.

End Lists.

Lemma map_fixed {A : Type} (f : A -> A) (l : list A) :
  (∀ a : A, f a = a) -> map f l = l.
Proof.
  intros Hf. induction l as [ | a l IH]; cbn [map]; [ reflexivity | ].
  rewrite Hf, IH. reflexivity.
Qed.

Theorem list_nominal : list_nominal_statement.
Proof.
  intros T N HN. constructor; simpl.
  - exact (list_equal_refl N HN).
  - exact (list_equal_sym N HN).
  - exact (list_equal_trans N HN).
  - intros ts. apply map_fixed, (apply_renaming_identity N HN).
  - intros second first ts. rewrite map_map. apply map_ext.
    exact (apply_renaming_compose N HN second first).
  - intros p p' ts Hsame. apply map_ext. intros t.
    exact (apply_renaming_same N HN p p' t Hsame).
  - exact (list_equal_equivariant N HN).
  - exact (list_free_names_equivariant N HN).
  - exact (list_free_names_equal N HN).
  - exact (list_apply_renaming_fixes N HN).
Qed.

(* ======================================================================== *)
(* The two sorts of renaming                                                *)
(* ======================================================================== *)

(* A renaming of continuations moves no variable, and one of variables
   no continuation; the two commute. *)
Lemma free_variables_continuations_renamed (p : permutation) (e : expr) :
  free_variables_of_expr (rename_expr (continuations_renaming p) e)
  = free_variables_of_expr e.
Proof.
  rewrite free_variables_of_expr_renamed. apply map_fixed. reflexivity.
Qed.

Lemma free_continuations_variables_renamed (p : permutation) (e : expr) :
  free_continuations_of_expr (rename_expr (variables_renaming p) e)
  = free_continuations_of_expr e.
Proof.
  rewrite free_continuations_of_expr_renamed. apply map_fixed. reflexivity.
Qed.

Lemma rename_sorts_commute (p kp : permutation) (e : expr) :
  rename_expr (variables_renaming p)
    (rename_expr (continuations_renaming kp) e)
  = rename_expr (continuations_renaming kp)
      (rename_expr (variables_renaming p) e).
Proof.
  rewrite <- !rename_expr_compose. apply rename_expr_ext.
  - intros x. unfold compose_renaming, rename_variable. cbn.
    rewrite app_nil_r. reflexivity.
  - intros k. unfold compose_renaming, rename_continuation. cbn.
    rewrite app_nil_r. reflexivity.
Qed.

(* ======================================================================== *)
(* A handler as a thing that contains continuations                         *)
(* ======================================================================== *)

Lemma handler_equal_equivariant (kp : permutation)
    (h h' : abstract_handler) :
  equal_abstraction list_of_names closed_expr_variables h h' ->
  equal_abstraction list_of_names closed_expr_variables
    (apply_renaming handler_continuations kp h)
    (apply_renaming handler_continuations kp h').
Proof.
  destruct h as [params handler], h' as [params' handler'].
  intros (p & p' & Hopens & Hopens' & Hbound & Hscope).
  exists p, p'.
  unfold opens, free_names_abstraction in *.
  simpl in Hopens, Hopens', Hbound, Hscope |- *.
  rewrite !rename_closed_expr_term, !free_variables_continuations_renamed.
  split; [ exact Hopens | ]. split; [ exact Hopens' | ].
  split; [ exact Hbound | ].
  unfold alpha_equivalent_closed in Hscope |- *.
  rewrite !rename_closed_expr_term in Hscope. rewrite !rename_closed_expr_term.
  rewrite !rename_sorts_commute.
  apply alpha_equivalent_equivariant, Hscope.
Qed.

Lemma handler_free_names_equal (h h' : abstract_handler) :
  equal_abstraction list_of_names closed_expr_variables h h' ->
  same_names (free_names handler_continuations h)
    (free_names handler_continuations h').
Proof.
  destruct h as [params handler], h' as [params' handler'].
  intros (p & p' & _ & _ & _ & Hscope). simpl in Hscope |- *.
  pose proof
    (free_names_equal closed_expr_continuations
       closed_expr_continuations_nominal _ _ Hscope)
    as Hsame.
  simpl in Hsame.
  rewrite !rename_closed_expr_term, !free_continuations_variables_renamed
    in Hsame.
  exact Hsame.
Qed.

Lemma handler_apply_renaming_fixes (kp : permutation)
    (h : abstract_handler) :
  fixes kp (free_names handler_continuations h) ->
  equal_abstraction list_of_names closed_expr_variables
    (apply_renaming handler_continuations kp h) h.
Proof.
  destruct h as [params handler]. simpl. intros Hfixes.
  exists [], [].
  split; [ intros name _; reflexivity | ].
  split; [ intros name _; reflexivity | ].
  split; [ reflexivity | ]. simpl.
  apply alpha_equivalent_closed_renamed.
  exact
    (apply_renaming_fixes closed_expr_continuations
       closed_expr_continuations_nominal kp handler Hfixes).
Qed.

Theorem handler_continuations_nominal
  : handler_continuations_nominal_statement.
Proof.
  pose proof
    (abstraction_nominal _ _ list_of_names closed_expr_variables
       list_of_names_bindable closed_expr_variables_nominal)
    as Habstraction.
  pose proof closed_expr_continuations_nominal as Hcontinuations.
  constructor.
  - exact (equal_refl _ Habstraction).
  - exact (equal_sym _ Habstraction).
  - exact (equal_trans _ Habstraction).
  - intros [params handler].
    exact (f_equal (create params)
             (apply_renaming_identity _ Hcontinuations handler)).
  - intros second first [params handler].
    exact (f_equal (create params)
             (apply_renaming_compose _ Hcontinuations second first handler)).
  - intros p p' [params handler] Hsame.
    exact (f_equal (create params)
             (apply_renaming_same _ Hcontinuations p p' handler Hsame)).
  - exact handler_equal_equivariant.
  - intros p [params handler] name.
    exact (free_names_equivariant _ Hcontinuations p handler name).
  - exact handler_free_names_equal.
  - exact handler_apply_renaming_fixes.
Qed.

Theorem group_continuations_nominal : group_continuations_nominal_statement.
Proof.
  apply pair_nominal;
    [ apply list_nominal, pair_nominal;
        [ apply no_names_nominal | exact handler_continuations_nominal ]
    | exact closed_expr_continuations_nominal ].
Qed.

(* ======================================================================== *)
(* The term of an abstraction                                               *)
(* ======================================================================== *)

Lemma group_handlers_closed (ks : list continuation) :
  ∀ hs : list group_handler,
    forallb (fun '(_, h) => handler_code_closed h)
      (combine ks (map handler_of hs))
    = true.
Proof.
  induction ks as [ | k ks IH]; intros [ | h hs]; cbn [map combine forallb];
    try reflexivity.
  rewrite IH, andb_true_r. exact (proj2_sig (scope (snd h))).
Qed.

Lemma map_fst_group (ks : list continuation) :
  ∀ hs : list group_handler,
    length ks = length hs -> map fst (combine ks (map handler_of hs)) = ks.
Proof.
  induction ks as [ | k ks IH]; intros [ | h hs] Hlength;
    cbn [map combine length fst] in Hlength |- *; try discriminate Hlength;
    [ reflexivity | ].
  injection Hlength as Hlength. rewrite (IH hs Hlength). reflexivity.
Qed.

Lemma group_free_continuations (ks : list continuation) :
  ∀ hs : list group_handler,
    length ks = length hs ->
    flat_map (fun '(_, h) => free_continuations_of_handler h)
      (combine ks (map handler_of hs))
    = flat_map (free_names group_handler_continuations) hs.
Proof.
  induction ks as [ | k ks IH]; intros [ | h hs] Hlength;
    cbn [map combine length flat_map] in Hlength |- *;
    try discriminate Hlength; [ reflexivity | ].
  injection Hlength as Hlength. rewrite (IH hs Hlength). reflexivity.
Qed.

Lemma handler_of_renamed (p : permutation) (h : group_handler) :
  rename_handler (continuations_renaming p) (handler_of h)
  = handler_of (apply_renaming group_handler_continuations p h).
Proof.
  destruct h as [metadata [params handler]]. unfold handler_of. simpl.
  rewrite rename_closed_expr_term. f_equal. apply map_fixed. reflexivity.
Qed.

(* The group renamed is the group of the abstraction renamed. *)
Lemma group_renamed (p : permutation) (ks : list continuation) :
  ∀ hs : list group_handler,
    rename_handlers (continuations_renaming p)
      (combine ks (map handler_of hs))
    = combine (map (permute p) ks)
        (map handler_of
           (map (apply_renaming group_handler_continuations p) hs)).
Proof.
  induction ks as [ | k ks IH]; intros [ | h hs]; try reflexivity.
  cbn [map combine]. rewrite <- IH, <- handler_of_renamed. reflexivity.
Qed.

(* One side, opened by a permutation that fixes the free names of the
   abstraction; [zs] is what the continuations become. *)
Lemma group_opened_abstraction (p : permutation) (ks zs : list continuation)
    (hs : list group_handler) (body : closed_expr) :
  length ks = length hs ->
  fixes p
    (without ks
       (flat_map (free_names group_handler_continuations) hs
        ++ free_continuations_of_expr (proj1_sig body))) ->
  map (permute p) ks = zs ->
  alpha_equivalent
    (Let_cont_recursive (combine ks (map handler_of hs)) (proj1_sig body))
    (Let_cont_recursive
       (combine zs
          (map handler_of
             (map (apply_renaming group_handler_continuations p) hs)))
       (rename_expr (continuations_renaming p) (proj1_sig body))).
Proof.
  intros Hlength Hfixes <-.
  assert (Hclosed :
    code_closed
      (Let_cont_recursive (combine ks (map handler_of hs)) (proj1_sig body))
    = true).
  { cbn [code_closed]. rewrite group_handlers_closed.
    exact (proj2_sig body). }
  rewrite <- group_renamed.
  apply (alpha_let_cont_recursive (continuations_renaming p) _ _ Hclosed).
  - intros y _. reflexivity.
  - intros k Hk. apply Hfixes.
    apply (cont_free_agrees k _ Hclosed) in Hk.
    cbn [free_continuations_of_expr] in Hk.
    rewrite (map_fst_group ks hs Hlength),
      (group_free_continuations ks hs Hlength)
      in Hk.
    exact Hk.
Qed.

(* A handler of a group, opened by a permutation that fixes its free
   variables; [params] is what the parameters become. *)
Lemma abstract_handler_opened (p : permutation)
    (before : list (continuation * continuation_handler)) (k : continuation)
    (h : abstract_handler) (metadata : handler_metadata)
    (after : list (continuation * continuation_handler)) (body : expr)
    (params : list variable) :
  fixes p (without (bound h) (free_variables_of_expr (proj1_sig (scope h))))
  ->
  map (permute p) (bound h) = params ->
  alpha_equivalent
    (Let_cont_recursive
       (before ++ (k, Handler (bound h) metadata (proj1_sig (scope h)))
        :: after)
       body)
    (Let_cont_recursive
       (before
        ++ (k,
            Handler params metadata
              (rename_expr (variables_renaming p) (proj1_sig (scope h))))
        :: after)
       body).
Proof.
  intros Hfixes <-.
  apply (Alpha_recursive_handler_by_relation (variables_renaming p) before k
           (bound h) metadata (proj1_sig (scope h)) after body).
  - exact (proj2_sig (scope h)).
  - intros y Hnotin Hy. apply Hfixes, among_without_list; [ exact Hnotin | ].
    apply closed_var_free, Hy.
  - reflexivity.
Qed.

Lemma abstract_handler_bound (h h' : abstract_handler) :
  equal_abstraction list_of_names closed_expr_variables h h' ->
  ∀ (before : list (continuation * continuation_handler)) (k : continuation)
    (metadata : handler_metadata)
    (after : list (continuation * continuation_handler)) (body : expr),
    alpha_equivalent
      (Let_cont_recursive
         (before ++ (k, Handler (bound h) metadata (proj1_sig (scope h)))
          :: after)
         body)
      (Let_cont_recursive
         (before ++ (k, Handler (bound h') metadata (proj1_sig (scope h')))
          :: after)
         body).
Proof.
  intros (p & p' & Hopens & Hopens' & Hbound & Hscope) before k metadata after
    body.
  unfold opens, free_names_abstraction in Hopens, Hopens'.
  simpl in Hopens, Hopens', Hbound, Hscope.
  unfold alpha_equivalent_closed in Hscope.
  rewrite !rename_closed_expr_term in Hscope.
  eapply Alpha_trans;
    [ exact (abstract_handler_opened p before k h metadata after body _
               Hopens Hbound)
    | ].
  eapply Alpha_trans;
    [ | apply Alpha_sym;
        exact (abstract_handler_opened p' before k h' metadata after body _
                 Hopens' eq_refl) ].
  exact
    (Alpha_context
       (Let_cont_recursive_handler before k (map (permute p') (bound h'))
          metadata Hole after body)
       _ _ Hscope).
Qed.

(* Two groups with the same continuations and equal handlers. [done]
   is what has been replaced. *)
Lemma group_same_names (body : expr) (hs hs' : list group_handler) :
  Forall2 (equal group_handler_continuations) hs hs' ->
  ∀ (zs : list continuation)
    (done : list (continuation * continuation_handler)),
    alpha_equivalent
      (Let_cont_recursive (done ++ combine zs (map handler_of hs)) body)
      (Let_cont_recursive (done ++ combine zs (map handler_of hs')) body).
Proof.
  induction 1 as [ | h h' hs hs' Hh _ IH]; intros zs done;
    (destruct zs as [ | z zs]; [ apply Alpha_refl | ]);
    [ apply Alpha_refl | ].
  cbn [map combine].
  apply Alpha_trans
    with (Let_cont_recursive
            (done ++ (z, handler_of h') :: combine zs (map handler_of hs))
            body).
  { destruct h as [metadata a], h' as [metadata' a'].
    destruct Hh as [Hmetadata Ha]. cbn [fst snd no_names equal] in Hmetadata.
    subst metadata'.
    exact (abstract_handler_bound a a' Ha done z metadata _ body). }
  replace (done ++ (z, handler_of h') :: combine zs (map handler_of hs))
    with ((done ++ [(z, handler_of h')]) ++ combine zs (map handler_of hs))
    by (rewrite <- app_assoc; reflexivity).
  replace (done ++ (z, handler_of h') :: combine zs (map handler_of hs'))
    with ((done ++ [(z, handler_of h')]) ++ combine zs (map handler_of hs'))
    by (rewrite <- app_assoc; reflexivity).
  apply IH.
Qed.

Theorem group_abstraction : group_abstraction_statement.
Proof.
  intros [ks [hs body]] [ks' [hs' body']] Hlength
    (p & p' & Hopens & Hopens' & Hbound & Hscope).
  unfold opens, free_names_abstraction in Hopens, Hopens'.
  simpl in Hlength, Hopens, Hopens', Hbound, Hscope.
  destruct Hscope as [Hhandlers Hbodies].
  unfold alpha_equivalent_closed in Hbodies.
  rewrite !rename_closed_expr_term in Hbodies.
  assert (Hlength' : length ks' = length hs').
  { pose proof (f_equal (@length _) Hbound) as Hnames.
    rewrite !length_map in Hnames.
    pose proof (Forall2_length Hhandlers) as Hscopes.
    rewrite !length_map in Hscopes.
    exact (eq_trans (eq_sym Hnames) (eq_trans Hlength Hscopes)). }
  unfold group_term. cbn [bound scope fst snd].
  eapply Alpha_trans;
    [ exact (group_opened_abstraction p ks _ hs body Hlength Hopens Hbound)
    | ].
  eapply Alpha_trans;
    [ | apply Alpha_sym;
        exact (group_opened_abstraction p' ks' _ hs' body' Hlength' Hopens'
                 eq_refl) ].
  apply Alpha_trans
    with (Let_cont_recursive
            (combine (map (permute p') ks')
               (map handler_of
                  (map (apply_renaming group_handler_continuations p) hs)))
            (rename_expr (continuations_renaming p') (proj1_sig body'))).
  - exact (Alpha_context (Let_cont_recursive_body _ Hole) _ _ Hbodies).
  - exact (group_same_names _ _ _ Hhandlers (map (permute p') ks') []).
Qed.

(* ======================================================================== *)
(* Alpha-equivalent groups are equal abstractions                           *)
(* ======================================================================== *)

(* Two handlers that agree at fresh parameters are equal. *)
Lemma alpha_handlers_equal (h h' : group_handler) :
  distinct (bound (snd h)) = true ->
  distinct (bound (snd h')) = true ->
  alpha_handlers (handler_of h) (handler_of h') ->
  equal group_handler_continuations h h'.
Proof.
  destruct h as [metadata [params handler]],
    h' as [metadata' [params' handler']].
  cbn [handler_of fst snd bound scope alpha_handlers].
  intros Hdistinct Hdistinct' (Hlength & Hmetadata & Hfresh).
  split; [ exact Hmetadata | ]. cbn [snd].
  destruct
    (for_fresh_names_elim _ _
       ((params ++ free_variables_of_expr (proj1_sig handler))
        ++ params' ++ free_variables_of_expr (proj1_sig handler'))
       Hfresh)
    as (zs & Hcount & Houtside & Hscopes).
  apply distinct_outside_app in Houtside as [Houtside Houtside'].
  pose proof Houtside as Hfree.
  apply distinct_outside_app in Hfree as [Hparams [_ Hfree]].
  pose proof Houtside' as Hfree'.
  apply distinct_outside_app in Hfree' as [Hparams' [_ Hfree']].
  exists (combine params zs), (combine params' zs).
  unfold opens, free_names_abstraction. simpl.
  split; [ | split; [ | split ] ].
  - intros name Hname. rewrite among_without in Hname.
    apply andb_true_iff in Hname as [Hname Hnot].
    apply negb_true_iff in Hnot.
    apply permute_combine_other; [ exact Hnot | ].
    destruct (among name zs) eqn:Hamong; [ | reflexivity ].
    apply Hfree in Hamong. congruence.
  - intros name Hname. rewrite among_without in Hname.
    apply andb_true_iff in Hname as [Hname Hnot].
    apply negb_true_iff in Hnot.
    apply permute_combine_other; [ exact Hnot | ].
    destruct (among name zs) eqn:Hamong; [ | reflexivity ].
    apply Hfree' in Hamong. congruence.
  - exact
      (eq_trans (permute_combine params zs (eq_sym Hcount) Hdistinct Hparams)
         (eq_sym
            (permute_combine params' zs
               (eq_sym (eq_trans Hcount Hlength)) Hdistinct' Hparams'))).
  - unfold alpha_equivalent_closed. rewrite !rename_closed_expr_term.
    exact Hscopes.
Qed.

Lemma alpha_handlers_all_equal (hs : list group_handler) :
  ∀ hs' : list group_handler,
    Forall (fun h => distinct (bound (snd h)) = true) hs ->
    Forall (fun h => distinct (bound (snd h)) = true) hs' ->
    Forall2 alpha_handlers (map handler_of hs) (map handler_of hs') ->
    Forall2 (equal group_handler_continuations) hs hs'.
Proof.
  induction hs as [ | h hs IH]; intros [ | h' hs'] Hdistinct Hdistinct' Hall;
    cbn [map] in Hall; inversion Hall as [ | ? ? ? ? Hfirst Hrest]; subst;
    [ constructor | ].
  inversion Hdistinct as [ | ? ? Hh Hhs]; subst.
  inversion Hdistinct' as [ | ? ? Hh' Hhs']; subst.
  constructor;
    [ exact (alpha_handlers_equal h h' Hh Hh' Hfirst)
    | exact (IH hs' Hhs Hhs' Hrest) ].
Qed.

(* The parameters of the handlers of a group whose binders are
   distinct. *)
Lemma group_params_distinct (ks : list continuation) :
  ∀ hs : list group_handler,
    length ks = length hs ->
    forallb (fun '(_, h) => handler_binders_distinct h)
      (combine ks (map handler_of hs))
    = true ->
    Forall (fun h => distinct (bound (snd h)) = true) hs.
Proof.
  induction ks as [ | k ks IH]; intros [ | h hs] Hlength Hall;
    cbn [length map combine forallb] in Hlength, Hall;
    try discriminate Hlength; constructor.
  - apply andb_true_iff in Hall as [Hfirst _].
    unfold handler_of in Hfirst. cbn [handler_binders_distinct] in Hfirst.
    apply andb_true_iff in Hfirst as [Hparams _]. exact Hparams.
  - injection Hlength as Hlength.
    apply andb_true_iff in Hall as [_ Hall]. exact (IH hs Hlength Hall).
Qed.

Lemma length_group (ks : list continuation) :
  ∀ hs : list group_handler,
    length ks = length hs ->
    length (combine ks (map handler_of hs)) = length ks.
Proof.
  induction ks as [ | k ks IH]; intros [ | h hs] Hlength;
    cbn [map combine length] in Hlength |- *; try discriminate Hlength;
    [ reflexivity | ].
  injection Hlength as Hlength. rewrite (IH hs Hlength). reflexivity.
Qed.

Lemma map_snd_combine {A B : Type} (l : list A) :
  ∀ l' : list B, length l = length l' -> map snd (combine l l') = l'.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] Hlength;
    cbn [length map combine snd] in Hlength |- *; try discriminate Hlength;
    [ reflexivity | ].
  injection Hlength as Hlength. rewrite (IH l' Hlength). reflexivity.
Qed.

(* What a group gives when its continuations are exchanged for new
   ones: the permutation opens the abstraction, the bound names become
   the new ones, and the handlers of the renamed term are those of the
   renamed scope. *)
Lemma group_swapped (ks zks : list continuation) (hs : list group_handler)
    (body : closed_expr) :
  length ks = length hs ->
  length zks = length ks ->
  distinct ks = true ->
  distinct_outside
    (ks
     ++ flat_map (free_names group_handler_continuations) hs
     ++ free_continuations_of_expr (proj1_sig body))
    zks ->
  fixes (combine ks zks)
    (without ks
       (flat_map (free_names group_handler_continuations) hs
        ++ free_continuations_of_expr (proj1_sig body)))
  ∧ map (permute (combine ks zks)) ks = zks
  ∧ map snd
      (rename_handlers (swap_continuations ks zks)
         (combine ks (map handler_of hs)))
    = map handler_of
        (map (apply_renaming group_handler_continuations (combine ks zks))
           hs).
Proof.
  intros Hlength Hcount Hdistinct Houtside.
  apply distinct_outside_app in Houtside as [Hnames [_ Hfree]].
  split; [ | split ].
  - intros name Hname. rewrite among_without in Hname.
    apply andb_true_iff in Hname as [Hname Hnot].
    apply negb_true_iff in Hnot.
    apply permute_combine_other; [ exact Hnot | ].
    destruct (among name zks) eqn:Hamong; [ | reflexivity ].
    apply Hfree in Hamong. congruence.
  - exact (permute_combine ks zks (eq_sym Hcount) Hdistinct Hnames).
  - change (swap_continuations ks zks)
      with (continuations_renaming (combine ks zks)).
    rewrite group_renamed. apply map_snd_combine.
    rewrite !length_map. exact Hlength.
Qed.

Theorem group_abstraction_iff : group_abstraction_iff_statement.
Proof.
  intros a a' Hlength Hlength' Hdistinct Hdistinct'.
  split; [ apply group_abstraction; exact Hlength | ].
  destruct a as [ks [hs body]], a' as [ks' [hs' body']].
  unfold group_term in *. cbn [bound scope fst snd] in *.
  intros Halpha.
  assert (Hclosed :
    code_closed
      (Let_cont_recursive (combine ks (map handler_of hs)) (proj1_sig body))
    = true).
  { cbn [code_closed]. rewrite group_handlers_closed.
    exact (proj2_sig body). }
  assert (Hclosed' :
    code_closed
      (Let_cont_recursive (combine ks' (map handler_of hs'))
         (proj1_sig body'))
    = true).
  { cbn [code_closed]. rewrite group_handlers_closed.
    exact (proj2_sig body'). }
  apply (alpha_let_cont_recursive_iff _ _ _ _ Hclosed Hclosed' Hdistinct
           Hdistinct')
    in Halpha.
  destruct Halpha as [_ Hfresh].
  (* The names and the parameters are distinct. *)
  pose proof (group_names_distinct _ _ Hdistinct) as Hnames.
  pose proof (group_names_distinct _ _ Hdistinct') as Hnames'.
  rewrite (map_fst_group ks hs Hlength) in Hnames.
  rewrite (map_fst_group ks' hs' Hlength') in Hnames'.
  cbn [binders_distinct] in Hdistinct, Hdistinct'.
  apply andb_true_iff in Hdistinct as [Hdistinct _].
  apply andb_true_iff in Hdistinct as [_ Hparams].
  apply andb_true_iff in Hdistinct' as [Hdistinct' _].
  apply andb_true_iff in Hdistinct' as [_ Hparams'].
  apply (group_params_distinct ks hs Hlength) in Hparams.
  apply (group_params_distinct ks' hs' Hlength') in Hparams'.
  (* Fresh continuations for both groups. *)
  destruct
    (for_fresh_names_elim _ _
       ((ks
         ++ flat_map (free_names group_handler_continuations) hs
         ++ free_continuations_of_expr (proj1_sig body))
        ++ ks'
        ++ flat_map (free_names group_handler_continuations) hs'
        ++ free_continuations_of_expr (proj1_sig body'))
       Hfresh)
    as (zks & Hcount & Houtside & Hhandlers & Hbodies).
  apply distinct_outside_app in Houtside as [Houtside Houtside'].
  rewrite (map_fst_group ks hs Hlength), (map_fst_group ks' hs' Hlength')
    in Hhandlers, Hbodies.
  assert (Hcount_ks : length zks = length ks).
  { exact (eq_trans Hcount (length_group ks hs Hlength)). }
  assert (Hsame : length ks = length ks').
  { pose proof (Forall2_length Hhandlers) as Hboth.
    rewrite !length_map in Hboth. unfold rename_handlers in Hboth.
    rewrite !length_map in Hboth.
    exact
      (eq_trans (eq_sym (length_group ks hs Hlength))
         (eq_trans Hboth (length_group ks' hs' Hlength'))). }
  destruct (group_swapped ks zks hs body Hlength Hcount_ks Hnames Houtside)
    as (Hopens & Hmoved & Hrenamed).
  destruct
    (group_swapped ks' zks hs' body' Hlength' (eq_trans Hcount_ks Hsame)
       Hnames' Houtside')
    as (Hopens' & Hmoved' & Hrenamed').
  rewrite Hrenamed, Hrenamed' in Hhandlers.
  exists (combine ks zks), (combine ks' zks).
  split; [ exact Hopens | ]. split; [ exact Hopens' | ].
  split; [ exact (eq_trans Hmoved (eq_sym Hmoved')) | ].
  split.
  - apply alpha_handlers_all_equal; [ | | exact Hhandlers ].
    + clear -Hparams. induction Hparams as [ | [metadata h] hs Hh _ IH];
        cbn [map]; constructor; [ exact Hh | exact IH ].
    + clear -Hparams'. induction Hparams' as [ | [metadata h] hs Hh _ IH];
        cbn [map]; constructor; [ exact Hh | exact IH ].
  - simpl. unfold alpha_equivalent_closed.
    rewrite !rename_closed_expr_term. exact Hbodies.
Qed.
