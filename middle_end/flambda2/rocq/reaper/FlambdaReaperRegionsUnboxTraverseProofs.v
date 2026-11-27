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

(* What the graph of a unit of the fragment of
   FlambdaReaperRegionsUnboxSpec.v has: its accessors read no field of the
   kind [Region], so no variable of an unboxed field is of that kind; the
   variable of each region that it begins has any source, so that no
   allocation dominates it in a solution closed under the rules; and the
   unit is of the fragment of FlambdaReaperFlowSpec.v wherever its erasure
   is.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsUnboxProofs.v. The traversal is that of
   FlambdaReaperUnboxingRefinesProofs.closure_free_reads_blocks, with the
   kinds of the fields read. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperRebuildSpec
  FlambdaReaperUnboxing FlambdaRegionsSpec FlambdaReaperRegionsSpec
  FlambdaReaperRegionsUnboxSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperUnboxingFuelSpec
  FlambdaReaperUnboxingFuelProofs FlambdaReaperSolutionProofs
  FlambdaReaperRegionsRebuildProofs FlambdaReaperRegionsTraverseProofs.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.
Module UF := FlambdaReaperUnboxingFuelSpec.
Module UFP := FlambdaReaperUnboxingFuelProofs.

(* ======================================================================== *)
(* The fields that the graph reads                                          *)
(* ======================================================================== *)

(* A field of a block or a boxed number, not of the kind [Region]. *)
Definition plain_read (f : field) : bool :=
  is_block_like_field f && kind_not_region (field_kind f).

Definition reads_plain (l : list fact) : Prop :=
  ∀ (to_ : node) (f : field) (base : node),
    Fact_accessor to_ f base ∈ l -> plain_read f = true.

Abbreviation rp a := (reads_plain (acc_facts a)).

Lemma rp_app (l l' : list fact) :
  reads_plain l -> reads_plain l' -> reads_plain (l ++ l').
Proof.
  intros H H' to_ f base Hin. apply in_app_or in Hin as [Hin | Hin];
    [ exact (H _ _ _ Hin) | exact (H' _ _ _ Hin) ].
Qed.

Lemma rp_add_facts (l : list fact) (a : acc) :
  reads_plain l -> rp a -> rp (add_facts l a).
Proof. intros Hl Ha. exact (rp_app _ _ Hl Ha). Qed.

Lemma rp_none (l : list fact) :
  (∀ f, f ∈ l -> match f with Fact_accessor _ _ _ => False | _ => True end) ->
  reads_plain l.
Proof. intros H to_ f base Hin. exact (False_rect _ (H _ Hin)). Qed.

Lemma rp_map {A : Type} (g : A -> fact) (l : list A) :
  (∀ a, match g a with Fact_accessor _ _ _ => False | _ => True end) ->
  reads_plain (map g l).
Proof.
  intros H. apply rp_none. intros f Hin.
  apply in_map_iff in Hin as (a & <- & _).
  apply H.
Qed.

Lemma rp_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  rp a -> rp (cond_any_usage env s a).
Proof.
  intros Ha. unfold cond_any_usage.
  destruct (env_current_code_id env); apply rp_add_facts;
    try (rewrite URP.note_simple_facts; exact Ha);
    apply rp_none; intros f [<- | []]; exact I.
Qed.

Lemma rp_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  rp a -> rp (snd (find_cont env k n a)).
Proof.
  intros Ha. unfold find_cont.
  destruct (lookup k (FlambdaReaperTraversal.env_conts env)); [ exact Ha | ].
  pose proof (URP.fresh_list_facts n a) as Hf.
  destruct (fresh_list n a) as [params a']. simpl in Hf |- *.
  apply rp_add_facts; [ apply rp_map; intros; exact I | ].
  rewrite Hf. exact Ha.
Qed.

Lemma rp_block_deps (base : node) (fields : list simple) (a : acc) :
  rp a -> rp (block_deps base fields a).
Proof.
  intros Ha. unfold block_deps. apply rp_add_facts.
  - apply rp_app; [ apply rp_map; intros [i s]; exact I | ].
    apply rp_none. intros f [<- | [<- | []]]; exact I.
  - rewrite URP.note_simples_facts. exact Ha.
Qed.

Lemma rp_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  rp a -> rp (traverse_prim env x p args a).
Proof.
  intros Ha. unfold traverse_prim.
  assert (Hacc : ∀ f s a, plain_read f = true -> rp a ->
            rp (add_fact (Fact_accessor (Node_variable x) f
                            (node_of_simple s)) (note_simple s a))).
  { intros f s a0 Hf Ha0. apply rp_add_facts.
    - intros to_ f' base [Heq | []]. injection Heq as _ <- _. exact Hf.
    - rewrite URP.note_simple_facts. exact Ha0. }
  assert (Hdefault :
    rp (add_facts
          (Fact_any_source (Node_variable x)
           :: map (fun from => Fact_use (Node_variable x) from)
                (free_names_of_simples args))
          (if prim_has_arbitrary_effects p
           then cond_any_usage env (Var x) a else a))).
  { apply rp_add_facts.
    - apply (rp_app [_]); [ apply rp_none; intros f [<- | []]; exact I | ].
      apply rp_map. intros; exact I.
    - destruct (prim_has_arbitrary_effects p);
        [ apply rp_cond_any_usage, Ha | exact Ha ]. }
  cbv zeta.
  destruct p as [p | p | p | p | p]; [ | exact Hdefault | | | ].
  - destruct p; destruct args as [ | s [ | s' args]]; try exact Hdefault;
      try destruct mut; try destruct variant_only; cbv beta iota;
      first
        [ exact Hdefault
        | apply Hacc; [ try destruct kind; reflexivity | exact Ha ]
        | apply rp_add_facts;
          [ apply rp_none; intros f [<- | []]; exact I | ];
          first
            [ apply Hacc; [ try destruct kind; reflexivity | exact Ha ]
            | rewrite URP.note_simple_facts; exact Ha ] ].
  - destruct p; [ apply rp_block_deps, Ha | exact Hdefault ].
  - destruct p; try exact Hdefault. apply rp_block_deps, Ha.
  - destruct p; try exact Hdefault. destruct args; exact Ha.
Qed.

Lemma rp_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  rp a -> rp (apply_cont_deps env ac a).
Proof.
  intros Ha. unfold apply_cont_deps.
  pose proof (rp_find_cont env (ac_continuation ac) (length (ac_args ac)) a Ha)
    as Hf.
  destruct (find_cont _ _ _ a) as [params a']. simpl in Hf.
  apply rp_add_facts; [ apply rp_map; intros [p s]; exact I | ].
  rewrite URP.note_simples_facts. exact Hf.
Qed.

Lemma rp_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  rp a -> rp (traverse_switch env scrutinee arms a).
Proof.
  intros Ha. unfold traverse_switch.
  pose proof (rp_cond_any_usage env scrutinee a Ha) as H.
  revert H. generalize (cond_any_usage env scrutinee a).
  induction (Keyed.compact Z.eqb arms) as [ | [d ac] l IH]; intros a' Ha';
    [ exact Ha' | ].
  simpl. apply IH. apply rp_apply_cont_deps, Ha'.
Qed.

Lemma rp_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  rp a -> rp (exn_handler_deps env h a).
Proof.
  intros Ha. unfold exn_handler_deps.
  destruct h as [[ | p params] metadata handler]; [ exact Ha | ].
  destruct (hm_is_exn_handler metadata); [ | exact Ha ].
  apply rp_cond_any_usage, rp_add_facts; [ | exact Ha ].
  apply rp_none. intros f [<- | []]. exact I.
Qed.

Lemma rp_traverse :
  (∀ e : expr, closure_free e = true ->
     ∀ (env : traverse_env) (a : acc), rp a -> rp (traverse env e a))
  ∧ (∀ h : continuation_handler, handler_closure_free h = true ->
       ∀ (env : traverse_env) (a : acc), rp a ->
         rp (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hd Hbody].
    change (traverse env (Let x kind d body) a)
      with (traverse env body (traverse_named env x d a)).
    apply (IH Hbody).
    destruct d as [s | p args | | | | | ]; try discriminate Hd.
    + apply rp_add_facts; [ apply rp_none; intros f [<- | []]; exact I | ].
      rewrite URP.note_simple_facts. exact Ha.
    + apply rp_traverse_prim, Ha.
  - intros bindings body _ IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hbs Hbody].
    change (traverse env (Let_static bindings body) a)
      with (traverse env body
              (List.fold_left (fun a b => traverse_binding env b a) bindings
                 (List.fold_left (fun a b => traverse_static_binding env b a)
                    bindings
                    (List.fold_left (fun a b => prepare_binding b a)
                       bindings a)))).
    apply (IH Hbody).
    rewrite forallb_forall in Hbs.
    assert (Hblocks : ∀ b, b ∈ bindings ->
              ∃ s tag fields,
                b = Bind_symbol s (Static_const_block tag fields)).
    { intros b Hb. specialize (Hbs b Hb).
      destruct b as [ | | s [ | | tag fields | | | ]]; try discriminate Hbs.
      eauto. }
    clear Hbs.
    assert (Hprep : ∀ a,
              List.fold_left (fun a b => prepare_binding b a) bindings a = a).
    { clear -Hblocks. induction bindings as [ | b bindings IHb]; intros a;
        [ reflexivity | ].
      destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
      simpl. apply IHb. intros b' Hb'. apply Hblocks. right. exact Hb'. }
    assert (Hcode : ∀ a,
              List.fold_left (fun a b => traverse_binding env b a) bindings a
              = a).
    { clear -Hblocks. induction bindings as [ | b bindings IHb]; intros a;
        [ reflexivity | ].
      destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
      simpl. apply IHb. intros b' Hb'. apply Hblocks. right. exact Hb'. }
    rewrite Hcode, Hprep. clear Hprep Hcode.
    revert a Ha. induction bindings as [ | b bindings IHb]; intros a Ha;
      [ exact Ha | ].
    destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
    simpl. apply IHb; [ intros b' Hb'; apply Hblocks; right; exact Hb' | ].
    apply rp_block_deps, Ha.
  - intros k h body IHh IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hh Hbody].
    change (traverse env (Let_cont_non_recursive k h body) a)
      with (traverse (add_cont env k (handler_params h)) body
              (exn_handler_deps env h (traverse_handler env h a))).
    apply (IH Hbody).
    apply rp_exn_handler_deps. exact (IHh Hh env a Ha).
  - intros handlers body IHh IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hhs Hbody].
    change (traverse env (Let_cont_recursive handlers body) a)
      with (traverse (add_rec_conts env handlers) body
              (List.fold_left
                 (fun a '(_, h) =>
                    traverse_handler (add_rec_conts env handlers) h a)
                 handlers a)).
    apply (IH Hbody).
    rewrite forallb_forall in Hhs.
    revert a Ha. generalize (add_rec_conts env handlers) as env'.
    intros env'. assert (Hsub : ∀ k h, (k, h) ∈ handlers -> (k, h) ∈ handlers)
      by auto. revert Hsub. generalize handlers at 1 3 as hs.
    intros hs Hsub. induction hs as [ | [k h] hs IHhs]; intros a Ha;
      [ exact Ha | ].
    simpl. apply IHhs; [ intros k' h' Hin; apply Hsub; right; exact Hin | ].
    assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
    exact (IHh k h Hin (Hhs _ Hin) env' a Ha).
  - intros ap Hcf. discriminate Hcf.
  - intros ac _ env a Ha. apply rp_apply_cont_deps, Ha.
  - intros scrutinee arms _ env a Ha. apply rp_traverse_switch, Ha.
  - intros _ env a Ha. exact Ha.
  - intros params metadata handler IH Hcf env a Ha. exact (IH Hcf env a Ha).
Qed.

Lemma rp_closure_deps (cds : finite_map code_dep) (ds : list closure_dep) :
  ∀ a : acc,
    rp a -> rp (List.fold_left (fun a d => add_closure_dep cds d a) ds a).
Proof.
  induction ds as [ | d ds IH]; intros a0 Ha0; [ exact Ha0 | ].
  simpl. apply IH. unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) cds) as [cd | ].
  - apply rp_add_facts; [ | exact Ha0 ].
    apply rp_none. intros f Hf.
    destruct Hf as [<- | [<- | Hf]]; [ exact I | exact I | ].
    destruct (cd_unknown_arity_call_witnesses cd); [ destruct Hf | ].
    destruct Hf as [<- | []]. exact I.
  - destruct (fresh a0) as [witness a1] eqn:Hfresh.
    apply rp_add_facts.
    + apply rp_none. intros f Hf.
      repeat (destruct Hf as [<- | Hf]; [ exact I | ]). destruct Hf.
    + unfold fresh in Hfresh. injection Hfresh as _ <-. exact Ha0.
Qed.

Lemma rp_resolve_delayed_deps (symbols : list symbol) (a : acc) :
  rp a -> rp (resolve_delayed_deps symbols a).
Proof.
  intros Ha. unfold resolve_delayed_deps.
  apply rp_add_facts; [ apply rp_map; intros; exact I | ].
  apply rp_closure_deps. apply rp_add_facts; [ | exact Ha ].
  apply rp_none. intros f Hf. apply in_flat_map in Hf as (d & _ & Hf).
  unfold apply_dep_facts in Hf.
  destruct (lookup _ _); [ unfold alias_for_caller in Hf | ];
    destruct (ad_function_containing_apply_expr d);
    repeat (destruct Hf as [<- | Hf]; [ exact I | ]); destruct Hf.
Qed.

Lemma rp_f_accessor (l : list fact) :
  reads_plain l ->
  ∀ (to_ : node) (f : field) (base : node),
    (to_, f, base) ∈ f_accessor (facts_of_list l) -> plain_read f = true.
Proof.
  intros Hl to_ f base Hin.
  cbn [facts_of_list f_accessor] in Hin.
  apply in_flat_map in Hin as (fct & Hfct & Hin).
  destruct fct; simpl in Hin; try contradiction.
  destruct Hin as [[= <- <- <-] | []]. exact (Hl _ _ _ Hfct).
Qed.

(* The graph of a unit without closures, projections, calls and code
   reads only fields of blocks and of boxed numbers, of no field of the
   kind [Region]. *)
Lemma closure_free_reads_plain (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  closure_free e = true ->
  ∀ (to_ : node) (f : field) (base : node),
    (to_, f, base) ∈ f_accessor (traverse_unit flag u e) ->
    kind_not_region (field_kind f) = true.
Proof.
  intros Hcf to_ f base Hin. unfold traverse_unit in Hin. cbv zeta in Hin.
  apply rp_f_accessor in Hin; [ apply andb_true_iff in Hin; apply Hin | ].
  apply rp_resolve_delayed_deps.
  apply rp_add_facts; [ apply rp_none; intros f' [<- | []]; exact I | ].
  apply (proj1 rp_traverse e Hcf).
  apply rp_add_facts; [ apply rp_none; intros f' [<- | []]; exact I | ].
  apply rp_add_facts; [ apply rp_none; intros f' [<- | []]; exact I | ].
  intros to' f' base' [].
Qed.

(* ======================================================================== *)
(* The kinds of the unboxed fields                                          *)
(* ======================================================================== *)

(* No field of an unboxing, at any depth, is of the kind [Region]. *)
Fixpoint ukinds_ok (u : unboxed_field) : bool :=
  match u with
  | Not_unboxed _ => true
  | Unboxed fields =>
    (fix go (l : unboxed_fields) : bool :=
       match l with
       | [] => true
       | (f, u) :: l => kind_not_region (field_kind f) && ukinds_ok u && go l
       end) fields
  end.

Lemma ukinds_ok_cons (f : field) (u : unboxed_field) (l : unboxed_fields) :
  ukinds_ok (Unboxed ((f, u) :: l))
  = kind_not_region (field_kind f) && ukinds_ok u && ukinds_ok (Unboxed l).
Proof. reflexivity. Qed.

Lemma leaves_plain :
  ∀ (u : unboxed_field) (kind : flambda_kind),
    kind_not_region kind = true -> ukinds_ok u = true ->
    forallb kind_not_region (map snd (unboxed_field_leaves kind u)) = true.
Proof.
  refine (URP.unboxed_field_rec _ _ _).
  - intros v kind Hk _. cbn. rewrite Hk. reflexivity.
  - intros fields IH kind _ Hok.
    induction fields as [ | [f u] fields IHl]; [ reflexivity | ].
    rewrite ukinds_ok_cons in Hok. apply andb_true_iff in Hok as [Hok Hrest].
    apply andb_true_iff in Hok as [Hf Hu].
    change (unboxed_field_leaves kind (Unboxed ((f, u) :: fields)))
      with (unboxed_field_leaves (field_kind f) u
            ++ unboxed_field_leaves kind (Unboxed fields)).
    rewrite map_app, forallb_app, (IH f u (or_introl eq_refl) _ Hf Hu).
    apply IHl; [ | exact Hrest ].
    intros f' u' Hin. apply (IH f' u'). right. exact Hin.
Qed.

Section Kinds.

Variables (F : facts) (T : tables) (cm : finite_map variable)
  (dominated : list (node * node)) (to_unbox : list node).

Hypothesis Hacc : ∀ (w : node) (f : field) (b : node),
  (w, f, b) ∈ f_accessor F -> kind_not_region (field_kind f) = true.

Lemma get_fields_kinds (uses : list node) :
  ∀ (f : field) (use : field_use),
    (f, use) ∈ get_fields F T uses -> kind_not_region (field_kind f) = true.
Proof.
  intros f use Hin.
  destruct (URP.get_fields_in F T uses f use Hin) as (w & b & Hw & _).
  exact (Hacc w f b Hw).
Qed.

Lemma field_step_kinds (fuel : nat) :
  (∀ fuel' : nat, fuel = S fuel' ->
     ∀ (block : node) (fields : list (field * field_use)) (n : positive),
       (∀ f use, (f, use) ∈ fields -> kind_not_region (field_kind f) = true) ->
       ukinds_ok (Unboxed (fst (mk_unboxed_fields F T cm dominated to_unbox
                                  fuel' block fields n)))
       = true) ->
  ∀ (block : node) (f : field) (use : field_use) (n : positive)
    (u : unboxed_field) (m : positive),
    URP.field_step F T cm dominated to_unbox fuel block f use n
    = (Some u, m) ->
    ukinds_ok u = true.
Proof.
  intros Hnest block f use n u m. unfold URP.field_step.
  destruct (negb (is_data_field f)); [ discriminate | ].
  destruct (get_single_field_source F T block f) as [ | source | ];
    [ discriminate | | ];
    destruct use as [ | flow_to ];
    try (intros H; injection H as <- _; reflexivity);
    destruct (forallb _ flow_to);
    try (intros H; injection H as <- _; reflexivity).
  destruct fuel as [ | fuel']; [ intros H; injection H as <- _; reflexivity | ].
  pose proof (Hnest fuel' eq_refl source
                (get_fields F T
                   (add_usages_through_function_slots F T cm true
                      (get_direct_usages T flow_to)))
                n (get_fields_kinds _)) as Hk.
  destruct (mk_unboxed_fields _ _ _ _ _ fuel' source _ n) as [nested m'].
  intros H. injection H as <- _. exact Hk.
Qed.

Lemma mk_unboxed_fields_kinds (fuel : nat) :
  ∀ (block : node) (fields : list (field * field_use)) (n : positive),
    (∀ f use, (f, use) ∈ fields -> kind_not_region (field_kind f) = true) ->
    ukinds_ok (Unboxed (fst (mk_unboxed_fields F T cm dominated to_unbox fuel
                               block fields n)))
    = true.
Proof.
  induction fuel as [ | fuel IHf]; intros block fields;
    [ assert (Hn : ∀ fuel', 0 = S fuel' ->
                 ∀ block fields n,
                   (∀ f use, (f, use) ∈ fields ->
                      kind_not_region (field_kind f) = true) ->
                   ukinds_ok (Unboxed (fst (mk_unboxed_fields F T cm dominated
                                              to_unbox fuel' block fields n)))
                   = true)
        by (intros ? H; discriminate H)
    | assert (Hn : ∀ fuel', S fuel = S fuel' ->
                 ∀ block fields n,
                   (∀ f use, (f, use) ∈ fields ->
                      kind_not_region (field_kind f) = true) ->
                   ukinds_ok (Unboxed (fst (mk_unboxed_fields F T cm dominated
                                              to_unbox fuel' block fields n)))
                   = true)
        by (intros ? H; injection H as <-; exact IHf) ];
    induction fields as [ | [f use] fields IH]; intros n Hf;
    rewrite ?URP.mk_unboxed_fields_nil; try reflexivity;
    rewrite URP.mk_unboxed_fields_cons;
    destruct (URP.field_step _ _ _ _ _ _ block f use n) as [u m] eqn:Hs;
    pose proof (IH m (fun f' use' Hin => Hf f' use' (or_intror Hin))) as Hr;
    destruct (mk_unboxed_fields _ _ _ _ _ _ block fields m) as [rest k];
    cbn [fst] in Hr |- *;
    (destruct u as [u | ]; [ | exact Hr ]);
    rewrite ukinds_ok_cons, (Hf f use (or_introl eq_refl)), Hr,
      (field_step_kinds _ Hn _ _ _ _ _ _ Hs);
    reflexivity.
Qed.

Lemma unboxed_fields_kinds (next : positive) (x : node)
    (fields : unboxed_fields) :
  (x, fields)
    ∈ fst (unboxed_fields_of_names F T cm dominated to_unbox next) ->
  forallb kind_not_region (map snd (fields_leaves fields)) = true.
Proof.
  intros Hin.
  destruct (URP.unboxed_fields_of_names_in F T cm dominated to_unbox next x
              fields Hin) as (a & next' & ->).
  apply leaves_plain; [ reflexivity | ].
  apply mk_unboxed_fields_kinds, get_fields_kinds.
Qed.

End Kinds.

(* ======================================================================== *)
(* The unit with its regions is of the fragment of the flow                 *)
(* ======================================================================== *)

Lemma flow_fragment_erase (codes codes' : list (code_id * code))
    (syms : list symbol) :
  (∀ e : expr, ∀ c : region_conts,
     closure_free e = true ->
     flow_fragment false codes' syms (erase c e) = true ->
     flow_fragment false codes syms e = true)
  ∧ (∀ h : continuation_handler, ∀ c : region_conts,
       handler_closure_free h = true ->
       handler_flow_fragment false codes' syms (erase_handler c h) = true ->
       handler_flow_fragment false codes syms h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c Hcf Hff. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hd Hb].
    destruct (FlambdaReaperRegionsRebuildProofs.region_prim_of d)
      as [[p args] | ] eqn:Hrp.
    + apply FlambdaReaperRegionsRebuildProofs.region_prim_of_some in Hrp.
      subst d. cbn [erase] in Hff. cbn [flow_fragment].
      destruct (erase_region_primitive x p args);
        [ cbn [flow_fragment] in Hff; apply andb_true_iff in Hff as [_ Hff]
        | ];
        exact (IH c Hb Hff).
    + rewrite (FlambdaReaperRegionsRebuildProofs.erase_let_plain _ _ _ _ _ Hrp)
        in Hff.
      cbn [flow_fragment] in Hff |- *. apply andb_true_iff in Hff as [Hn Hff].
      rewrite Hn. exact (IH c Hb Hff).
  - intros bindings body _ IH c Hcf Hff. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hbs Hb].
    cbn [erase flow_fragment] in Hff |- *.
    apply andb_true_iff in Hff as [_ Hff]. rewrite (IH c Hb Hff), andb_true_r.
    apply forallb_forall. intros bd Hin. rewrite forallb_forall in Hbs.
    specialize (Hbs bd Hin).
    destruct bd as [ | | s [ | | tag fields | | | ]]; try discriminate Hbs.
    reflexivity.
  - intros k h body IHh IH c Hcf Hff. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hh Hb].
    cbn [erase flow_fragment] in Hff |- *.
    apply andb_true_iff in Hff as [Hfh Hff].
    rewrite (IHh c Hh Hfh), (IH _ Hb Hff). reflexivity.
  - intros handlers body IHh IH c Hcf Hff. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hhs Hb].
    cbn [erase flow_fragment] in Hff |- *.
    apply andb_true_iff in Hff as [Hfh Hff].
    rewrite (IH _ Hb Hff), andb_true_r.
    set (c' := map (fun '(k, h) => (k, region_params h)) handlers ++ c)
      in Hff, Hfh.
    rewrite forallb_forall in Hhs, Hfh |- *. intros [k h] Hin.
    apply (IHh k h Hin c' (Hhs _ Hin)).
    apply (Hfh (k, erase_handler c' h)). apply in_map_iff.
    exists (k, h). split; [ reflexivity | exact Hin ].
  - intros ap c Hcf. discriminate Hcf.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH c Hcf Hff. cbn [erase_handler] in Hff.
    cbn [handler_flow_fragment] in Hff |- *. exact (IH c Hcf Hff).
Qed.

Lemma in_flow_fragment_of_erase (e : expr) :
  closure_free e = true -> in_flow_fragment (erase [] e) = true ->
  in_flow_fragment e = true.
Proof.
  intros Hcf H. unfold in_flow_fragment in H |- *.
  rewrite (proj1 (FlambdaReaperRegionsTraverseProofs.static_names_erase
                    symbols_of_binding ltac:(intros [ | | ]; reflexivity))
             e []) in H.
  rewrite (proj1 (FlambdaReaperRegionsTraverseProofs.static_names_erase
                    code_ids_of_binding ltac:(intros [ | | ]; reflexivity))
             e []) in H.
  apply andb_true_iff in H as [Hf Hd]. rewrite Hd, andb_true_r.
  exact (proj1 (flow_fragment_erase _ _ _) e [] Hcf Hf).
Qed.

(* ======================================================================== *)
(* The variables of the regions have any source                             *)
(* ======================================================================== *)

Lemma begins_any_source (G : graph) (noted : symbol -> Prop)
    (deps : closure_dep -> Prop) (cds : finite_map code_dep)
    (keep : value_slot -> bool) (codes : list (code_id * code))
    (syms : list symbol) (slots heap : Prop) :
  (∀ e : expr, ∀ (ctx : option code_id) (ce : finite_map (list node)),
     closure_free e = true ->
     FlambdaReaperFlowProofs.expr_edges G noted deps cds keep codes syms slots
       heap ctx ce e ->
     ∀ r : variable, r ∈ region_binders e -> g_any_source G (Node_variable r))
  ∧ (∀ h : continuation_handler,
       ∀ (ctx : option code_id) (ce : finite_map (list node)),
         handler_closure_free h = true ->
         FlambdaReaperFlowProofs.handler_edges G noted deps cds keep codes syms
           slots heap ctx ce h ->
         ∀ r : variable, r ∈ handler_region_binders h ->
           g_any_source G (Node_variable r)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH ctx ce Hcf He r Hr. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hd Hb].
    cbn [FlambdaReaperFlowProofs.expr_edges] in He. destruct He as [Hn He].
    destruct d as [s | [p | p | p | p | p] args | | | | | ];
      cbn [region_binders] in Hr;
      try exact (IH ctx ce Hb He r Hr).
    destruct p; cbn [region_binders] in Hr;
      try exact (IH ctx ce Hb He r Hr);
      (destruct Hr as [<- | Hr]; [ | exact (IH ctx ce Hb He r Hr) ]);
      cbn [FlambdaReaperFlowProofs.named_edges
             FlambdaReaperFlowProofs.prim_edges
             FlambdaReaperFlowProofs.prim_classified] in Hn;
      destruct Hn as [_ [Hs _]]; exact Hs.
  - intros bindings body _ IH ctx ce Hcf He r Hr. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hbs Hb].
    cbn [FlambdaReaperFlowProofs.expr_edges] in He. destruct He as [_ He].
    cbn [region_binders] in Hr. apply in_app_or in Hr as [Hr | Hr];
      [ | exact (IH ctx ce Hb He r Hr) ].
    exfalso. apply in_flat_map in Hr as (bd & Hin & Hr).
    rewrite forallb_forall in Hbs. specialize (Hbs bd Hin).
    destruct bd as [ | | s [ | | tag fields | | | ]]; try discriminate Hbs.
    destruct Hr.
  - intros k h body IHh IH ctx ce Hcf He r Hr. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hh Hb].
    cbn [FlambdaReaperFlowProofs.expr_edges] in He. destruct He as [Hhe He].
    cbn [region_binders] in Hr. apply in_app_or in Hr as [Hr | Hr];
      [ exact (IHh ctx ce Hh Hhe r Hr) | exact (IH ctx _ Hb He r Hr) ].
  - intros handlers body IHh IH ctx ce Hcf He r Hr.
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hhs Hb].
    cbn [FlambdaReaperFlowProofs.expr_edges] in He. destruct He as [Hhe He].
    cbn [region_binders] in Hr. apply in_app_or in Hr as [Hr | Hr];
      [ | exact (IH ctx _ Hb He r Hr) ].
    apply in_flat_map in Hr as ([k h] & Hin & Hr).
    rewrite forallb_forall in Hhs.
    apply (proj1 (every_in _ _ _)) with (a := (k, h)) in Hhe;
      [ | exact Hin ].
    exact (IHh k h Hin ctx _ (Hhs _ Hin) Hhe r Hr).
  - intros ap ctx ce Hcf. discriminate Hcf.
  - intros ac ctx ce _ _ r [].
  - intros scrutinee arms ctx ce _ _ r [].
  - intros ctx ce _ _ r [].
  - intros params md hbody IH ctx ce Hcf He r Hr.
    exact (IH ctx ce Hcf He r Hr).
Qed.
